"""
WiFi Manager module for handling WiFi connections on Linux devices.
Uses NetworkManager through DBus interface or direct wpa_supplicant commands
as a fallback.
"""
import os
import subprocess
import re
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Union

# Define the NetworkManager check function first before using it
def is_networkmanager_available() -> bool:
    """Check if NetworkManager is available on the system."""
    try:
        result = subprocess.run(['nmcli', '--version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Set up logging after defining functions it might use
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wifi_manager")

# Add file handler to save logs
file_handler = logging.FileHandler('/tmp/wifi_manager.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Now it's safe to use the function
logger.info("======= WiFi Manager Module Initialized =======")
logger.info(f"NetworkManager available: {is_networkmanager_available()}")

# Check if NetworkManager is available
def is_networkmanager_available() -> bool:
    """Check if NetworkManager is available on the system."""
    try:
        result = subprocess.run(['nmcli', '--version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

# Determine which method to use
USE_NETWORKMANAGER = is_networkmanager_available()

def scan_networks() -> List[Dict[str, Union[str, int]]]:
    """
    Scan for available WiFi networks.
    
    Returns:
        A list of dictionaries containing network information:
        [{'ssid': 'NetworkName', 'signal': -70, 'security': 'WPA2'}]
    """
    networks = []
    
    try:
        if USE_NETWORKMANAGER:
            # Use NetworkManager to scan
            logger.info("Scanning networks using NetworkManager")
            output = subprocess.check_output(
                ['nmcli', '-t', '-f', 'SSID,SIGNAL,SECURITY', 'device', 'wifi', 'list', '--rescan', 'yes'],
                text=True
            )
            
            for line in output.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.split(':')
                if len(parts) >= 3:
                    ssid = parts[0]
                    # Skip hidden networks
                    if not ssid:
                        continue
                        
                    try:
                        signal = int(parts[1])
                    except ValueError:
                        signal = 0
                        
                    security = parts[2] if len(parts) > 2 and parts[2] else "Open"
                    
                    # Avoid duplicate SSIDs, keep the one with strongest signal
                    existing_network = next((n for n in networks if n['ssid'] == ssid), None)
                    if existing_network:
                        if signal > existing_network['signal']:
                            existing_network['signal'] = signal
                            existing_network['security'] = security
                    else:
                        networks.append({
                            'ssid': ssid,
                            'signal': signal,
                            'security': security
                        })
        else:
            # Fallback to iwlist
            logger.info("Scanning networks using iwlist")
            output = subprocess.check_output(
                ['sudo', 'iwlist', 'wlan0', 'scan'],
                text=True
            )
            
            current_network = {}
            for line in output.split('\n'):
                line = line.strip()
                
                # Extract SSID
                ssid_match = re.search(r'ESSID:"([^"]*)"', line)
                if ssid_match:
                    if current_network and 'ssid' in current_network:
                        networks.append(current_network)
                    current_network = {'ssid': ssid_match.group(1)}
                
                # Extract signal strength
                signal_match = re.search(r'Signal level=(-\d+) dBm', line)
                if signal_match and 'ssid' in current_network:
                    current_network['signal'] = int(signal_match.group(1))
                
                # Extract encryption
                if 'Encryption key:on' in line and 'ssid' in current_network:
                    current_network['security'] = 'Protected'
                elif 'Encryption key:off' in line and 'ssid' in current_network:
                    current_network['security'] = 'Open'
            
            # Add the last network if there is one
            if current_network and 'ssid' in current_network:
                networks.append(current_network)
        
        # Sort by signal strength (strongest first)
        return sorted(networks, key=lambda x: x.get('signal', -100), reverse=True)
    
    except Exception as e:
        logger.error(f"Error scanning networks: {e}")
        return []

def connect_to_network(ssid: str, password: str) -> Tuple[bool, str]:
    """
    Connect to a WiFi network with the given SSID and password.
    
    Args:
        ssid: The network name
        password: The network password
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        if USE_NETWORKMANAGER:
            logger.info(f"Connecting to network {ssid} using NetworkManager")
            # First, check if connection profile already exists
            check_cmd = ['nmcli', '-t', '-f', 'NAME', 'connection', 'show']
            output = subprocess.check_output(check_cmd, text=True)
            
            connection_exists = False
            for line in output.strip().split('\n'):
                if line == ssid:
                    connection_exists = True
                    break
            
            if connection_exists:
                # Delete existing connection to ensure we use the new password
                logger.info(f"Deleting existing connection for {ssid}")
                try:
                    subprocess.run(['nmcli', 'connection', 'delete', ssid], check=False)
                except:
                    # Try with sudo if regular command fails
                    logger.info("Trying delete with sudo")
                    subprocess.run(['sudo', 'nmcli', 'connection', 'delete', ssid], check=False)
            
            # Connect to the network
            connect_cmd = [
                'nmcli', 'device', 'wifi', 'connect', ssid, 
                'password', password
            ]
            
            logger.info(f"Attempting to connect to {ssid}")
            result = subprocess.run(
                connect_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Connection successful")
                return True, "Successfully connected"
            else:
                logger.warning(f"Regular connect failed: {result.stderr.strip()}")
                
                # Try with sudo if regular command fails
                logger.info("Trying connect with sudo")
                sudo_cmd = ['sudo'] + connect_cmd
                sudo_result = subprocess.run(
                    sudo_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if sudo_result.returncode == 0:
                    logger.info("Connection with sudo successful")
                    return True, "Successfully connected"
                else:
                    logger.error(f"Connect with sudo also failed: {sudo_result.stderr.strip()}")
                    return False, f"Connection failed: {sudo_result.stderr.strip()}"
        else:
            # Fallback to wpa_supplicant
            logger.info(f"Connecting to network {ssid} using wpa_supplicant")
            
            # Generate wpa_supplicant configuration
            wpa_config = f'''
network={{
    ssid="{ssid}"
    psk="{password}"
}}
'''
            # Write configuration to temporary file
            config_file = "/tmp/wpa_supplicant.conf"
            with open(config_file, 'w') as f:
                f.write(wpa_config)
            
            # Stop any existing wpa_supplicant process
            subprocess.run(['sudo', 'killall', 'wpa_supplicant'], 
                          check=False)
            
            # Start wpa_supplicant with new configuration
            subprocess.run(['sudo', 'wpa_supplicant', '-B', '-i', 'wlan0', 
                           '-c', config_file], check=True)
            
            # Get IP address using DHCP
            subprocess.run(['sudo', 'dhclient', 'wlan0'], check=True)
            
            # Verify connection by checking if we have an IP
            time.sleep(5)  # Give it some time to connect
            ip_check = subprocess.check_output(['ip', 'addr', 'show', 'wlan0'], 
                                              text=True)
            
            if re.search(r'inet \d+\.\d+\.\d+\.\d+', ip_check):
                return True, "Successfully connected"
            else:
                return False, "Failed to obtain IP address"
                
    except subprocess.CalledProcessError as e:
        error_message = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
        logger.error(f"Error connecting to network: {error_message}")
        return False, f"Error: {error_message}"
    except Exception as e:
        logger.error(f"Unexpected error connecting to network: {e}")
        return False, f"Unexpected error: {str(e)}"

def get_connection_status() -> Dict[str, Union[bool, str]]:
    """
    Get the current WiFi connection status.
    
    Returns:
        A dictionary with connection information:
        {
            'connected': True/False,
            'ssid': 'NetworkName',
            'ip_address': '192.168.1.100'
        }
    """
    logger.info("=== Starting get_connection_status() ===")
    status = {
        'connected': False,
        'ssid': '',
        'ip_address': ''
    }
    
    try:
        if USE_NETWORKMANAGER:
            # Get connection status from NetworkManager
            output = subprocess.check_output(
                ['nmcli', '-t', '-f', 'DEVICE,STATE,CONNECTION', 'device'],
                text=True
            )
            
            wifi_connection = None
            for line in output.strip().split('\n'):
                parts = line.split(':')
                if len(parts) >= 3 and parts[0].startswith('wlan'):
                    if parts[1] == 'connected':
                        wifi_connection = parts[2]
                        status['connected'] = True
                        status['ssid'] = wifi_connection
                        break
            
            # If connected, get IP address
            if status['connected']:
                ip_output = subprocess.check_output(
                    ['ip', '-f', 'inet', 'addr', 'show', parts[0]],
                    text=True
                )
                ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', ip_output)
                if ip_match:
                    status['ip_address'] = ip_match.group(1)
                
        else:
            # Fallback to iwconfig and ifconfig
            try:
                iwconfig_output = subprocess.check_output(
                    ['iwconfig', 'wlan0'],
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                ssid_match = re.search(r'ESSID:"([^"]*)"', iwconfig_output)
                if ssid_match and ssid_match.group(1) != "off/any":
                    status['connected'] = True
                    status['ssid'] = ssid_match.group(1)
                    
                    # Get IP address
                    ip_output = subprocess.check_output(
                        ['ifconfig', 'wlan0'],
                        text=True
                    )
                    ip_match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', ip_output)
                    if ip_match:
                        status['ip_address'] = ip_match.group(1)
            except subprocess.CalledProcessError:
                pass
                
        logger.debug(f"Returning connection status: {status}")
        return status
    except Exception as e:
        logger.error(f"Error getting connection status: {e}", exc_info=True)
        return status
    finally:
        logger.info("=== Completed get_connection_status() ===")

def disconnect_from_network() -> Tuple[bool, str]:
    """
    Disconnect from the current WiFi network.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        logger.info("=== Starting disconnect_from_network() ===")
        if USE_NETWORKMANAGER:
            logger.info("Disconnecting from network using NetworkManager")
            
            # First check if we are connected to any wifi network
            status = get_connection_status()
            logger.debug(f"Current connection status: {status}")
            if not status['connected']:
                logger.info("Not currently connected to any network")
                return True, "Not connected to any network"
                
            # Get the current connection device
            logger.info("Fetching current WiFi device information")
            cmd = ['nmcli', '-t', '-f', 'DEVICE,STATE,CONNECTION', 'device']
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            output = subprocess.check_output(cmd, text=True)
            logger.debug(f"Command output: {output}")
            
            wifi_device = None
            for line in output.strip().split('\n'):
                logger.debug(f"Parsing device line: {line}")
                parts = line.split(':')
                if len(parts) >= 3 and parts[0].startswith('wlan'):
                    logger.debug(f"Found WiFi device: {parts[0]}, state: {parts[1]}")
                    if parts[1] == 'connected':
                        wifi_device = parts[0]
                        logger.info(f"Found connected WiFi device: {wifi_device}")
                        break
            
            if wifi_device:
                # Disconnect the device - try with sudo if needed
                logger.info(f"Attempting to disconnect device: {wifi_device}")
                try:
                    cmd = ['nmcli', 'device', 'disconnect', wifi_device]
                    logger.debug(f"Running command: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    logger.debug(f"Command stdout: {result.stdout}")
                    logger.info("Successfully disconnected without sudo")
                    return True, "Successfully disconnected from network"
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Regular disconnect failed: {e}")
                    logger.debug(f"Error output: {e.stderr}")
                    
                    # Try with sudo if regular command fails
                    logger.info("Trying disconnect with sudo")
                    sudo_cmd = ['sudo', 'nmcli', 'device', 'disconnect', wifi_device]
                    logger.debug(f"Running command: {' '.join(sudo_cmd)}")
                    
                    result = subprocess.run(
                        sudo_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    logger.debug(f"Command return code: {result.returncode}")
                    logger.debug(f"Command stdout: {result.stdout}")
                    logger.debug(f"Command stderr: {result.stderr}")
                    
                    if result.returncode == 0:
                        logger.info("Successfully disconnected with sudo")
                        return True, "Successfully disconnected from network"
                    else:
                        logger.error(f"Sudo disconnect failed: {result.stderr.strip()}")
                        return False, f"Failed to disconnect: {result.stderr.strip()}"
            else:
                logger.warning("No connected WiFi device found")
                return False, "No connected WiFi device found"
        else:
            # Fallback to direct commands
            logger.info("Disconnecting from network using direct commands")
            
            # Stop wpa_supplicant
            logger.info("Stopping wpa_supplicant")
            cmd = ['sudo', 'killall', 'wpa_supplicant']
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, 
                          check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True)
            logger.debug(f"Command return code: {result.returncode}")
            logger.debug(f"Command stderr: {result.stderr}")
            
            # Release DHCP lease
            logger.info("Releasing DHCP lease")
            cmd = ['sudo', 'dhclient', '-r', 'wlan0']
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, 
                          check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True)
            logger.debug(f"Command return code: {result.returncode}")
            logger.debug(f"Command stderr: {result.stderr}")
            
            # Verify disconnection
            logger.info("Waiting for disconnection to complete")
            time.sleep(2)  # Give it time to disconnect
            
            status = get_connection_status()
            logger.debug(f"Status after disconnect attempt: {status}")
            
            if not status['connected']:
                logger.info("Successfully disconnected - verified")
                return True, "Successfully disconnected from network"
            else:
                logger.error(f"Still connected after disconnect attempt: {status}")
                return False, "Failed to disconnect from network"
                
    except Exception as e:
        logger.error(f"Error disconnecting from network: {e}", exc_info=True)
        return False, f"Error: {str(e)}"
    finally:
        logger.info("=== Completed disconnect_from_network() ===")