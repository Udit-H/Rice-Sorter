// Variables for keyboard state
let currentInput = null;
let isShiftActive = false;
let isSpecialActive = false;

// Check current connection on page load
window.onload = function() {
    checkCurrentConnection();
    setupKeyboard();
};

// Original WiFi functionality
function checkCurrentConnection() {
    fetch('/wifi/status')
        .then(response => response.json())
        .then(data => {
            if(data.connected) {
                document.getElementById('current-connection').classList.remove('hidden');
                document.getElementById('current-ssid').textContent = data.ssid;
                document.getElementById('current-ip').textContent = data.ip_address;
            }
        })
        .catch(error => console.error('Error checking connection:', error));
}

function scanNetworks() {
    document.getElementById('scanning-status').classList.remove('hidden');
    document.getElementById('scan-button').disabled = true;
    document.getElementById('network-list-container').classList.add('hidden');
    
    fetch('/wifi/scan')
        .then(response => response.json())
        .then(data => {
            document.getElementById('scanning-status').classList.add('hidden');
            document.getElementById('scan-button').disabled = false;
            
            if(data.success) {
                populateNetworks(data.networks);
                document.getElementById('network-list-container').classList.remove('hidden');
            } else {
                showStatus('error', 'Failed to scan for networks: ' + data.error);
            }
        })
        .catch(error => {
            document.getElementById('scanning-status').classList.add('hidden');
            document.getElementById('scan-button').disabled = false;
            showStatus('error', 'Error scanning networks: ' + error);
        });
}

function populateNetworks(networks) {
    const networkList = document.getElementById('network-list');
    networkList.innerHTML = '';
    
    if(networks.length === 0) {
        networkList.innerHTML = '<p>No networks found.</p>';
        return;
    }
    
    networks.forEach(network => {
        const networkItem = document.createElement('div');
        networkItem.className = 'network-item';
        networkItem.onclick = function() {
            selectNetwork(network.ssid);
        };
        
        // Add signal strength indicator
        const signalStrength = getSignalStrengthText(network.signal);
        
        networkItem.innerHTML = `
            ${network.ssid}
            <span class="signal-strength">${signalStrength}</span>
        `;
        
        networkList.appendChild(networkItem);
    });
}

function getSignalStrengthText(signal) {
    if(signal >= -50) return '★★★★★';
    if(signal >= -60) return '★★★★☆';
    if(signal >= -70) return '★★★☆☆';
    if(signal >= -80) return '★★☆☆☆';
    return '★☆☆☆☆';
}

function selectNetwork(ssid) {
    const networkItems = document.querySelectorAll('.network-item');
    networkItems.forEach(item => {
        item.classList.remove('selected');
        if(item.textContent.trim().startsWith(ssid)) {
            item.classList.add('selected');
        }
    });
    
    document.getElementById('ssid').value = ssid;
    document.getElementById('wifi-form').classList.remove('hidden');
}

function setupFormSubmission() {
    document.getElementById('wifi-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const ssid = document.getElementById('ssid').value;
        const password = document.getElementById('password').value;
        
        // Display connecting status
        showStatus('info', '<span class="spinner"></span> Connecting to ' + ssid + '...');
        
        fetch('/wifi/connect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ssid: ssid,
                password: password
            })
        })
        .then(response => response.json())
        .then(data => {
            if(data.success) {
                showStatus('success', 'Successfully connected to ' + ssid);
                setTimeout(() => {
                    checkCurrentConnection();
                }, 2000);
            } else {
                showStatus('error', 'Failed to connect: ' + (data.error || data.message || "Unknown error"));
            }
        })
        .catch(error => {
            showStatus('error', 'Error connecting to network: ' + error);
        });
    });
}

function showStatus(type, message) {
    const statusElement = document.getElementById('connection-status');
    statusElement.className = 'status ' + type;
    statusElement.innerHTML = message;
    statusElement.classList.remove('hidden');
    
    // Auto-hide success messages after 5 seconds
    if(type === 'success') {
        setTimeout(() => {
            statusElement.classList.add('hidden');
        }, 5000);
    }
}

function disconnectWifi() {
    // Show disconnecting status
    showStatus('info', '<span class="spinner"></span> Disconnecting from network...');
    
    fetch('/wifi/disconnect', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if(data.success) {
            showStatus('success', 'Successfully disconnected from network');
            // Hide current connection info
            document.getElementById('current-connection').classList.add('hidden');
            // Refresh scan to show available networks
            setTimeout(() => {
                scanNetworks();
            }, 2000);
        } else {
            showStatus('error', 'Failed to disconnect: ' + (data.error || data.message));
        }
    })
    .catch(error => {
        showStatus('error', 'Error disconnecting from network: ' + error);
    });
}

// Virtual Keyboard functionality
function setupKeyboard() {
    // Setup password field focus event
    const passwordField = document.getElementById('password');
    passwordField.addEventListener('focus', function() {
        currentInput = this;
        showKeyboard();
    });
    
    // Setup shift and special keys
    document.getElementById('shift-key').addEventListener('click', function() {
        isShiftActive = !isShiftActive;
        isSpecialActive = false;
        renderKeyboard();
    });
    
    document.getElementById('special-key').addEventListener('click', function() {
        isSpecialActive = !isSpecialActive;
        isShiftActive = false;
        renderKeyboard();
    });
    
    // Initial keyboard setup
    renderKeyboard();
    
    // Setup form submission
    setupFormSubmission();
}

function renderKeyboard() {
    const layout = getKeyboardLayout();
    
    document.getElementById('row1').innerHTML = '';
    document.getElementById('row2').innerHTML = '';
    document.getElementById('row3').innerHTML = '';
    
    layout.row1.forEach(key => {
        appendKey(document.getElementById('row1'), key);
    });
    
    layout.row2.forEach(key => {
        appendKey(document.getElementById('row2'), key);
    });
    
    layout.row3.forEach(key => {
        appendKey(document.getElementById('row3'), key);
    });
}

function getKeyboardLayout() {
    if (isSpecialActive) {
        return {
            row1: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            row2: ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')'],
            row3: ['-', '_', '+', '=', '/', '\\', '[', ']', '{', '}']
        };
    } else {
        const lowercase = {
            row1: ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            row2: ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
            row3: ['z', 'x', 'c', 'v', 'b', 'n', 'm', '.', ',']
        };
        
        if (isShiftActive) {
            return {
                row1: lowercase.row1.map(c => c.toUpperCase()),
                row2: lowercase.row2.map(c => c.toUpperCase()),
                row3: lowercase.row3.map(c => c.toUpperCase())
            };
        }
        
        return lowercase;
    }
}

function appendKey(row, key) {
    const keyElement = document.createElement('div');
    keyElement.className = 'keyboard-key';
    keyElement.textContent = key;
    keyElement.onclick = function() { typeKey(key); };
    row.appendChild(keyElement);
}

function typeKey(key) {
    if (!currentInput) return;
    
    const start = currentInput.selectionStart;
    const end = currentInput.selectionEnd;
    const value = currentInput.value;
    
    currentInput.value = value.substring(0, start) + key + value.substring(end);
    currentInput.selectionStart = currentInput.selectionEnd = start + 1;
    currentInput.focus();
    
    // Reset shift after one character
    if (isShiftActive) {
        isShiftActive = false;
        renderKeyboard();
    }
}

function deleteChar() {
    if (!currentInput) return;
    
    const start = currentInput.selectionStart;
    const end = currentInput.selectionEnd;
    const value = currentInput.value;
    
    if (start === end && start > 0) {
        // No selection, delete one character before cursor
        currentInput.value = value.substring(0, start - 1) + value.substring(end);
        currentInput.selectionStart = currentInput.selectionEnd = start - 1;
    } else if (start !== end) {
        // Delete selection
        currentInput.value = value.substring(0, start) + value.substring(end);
        currentInput.selectionStart = currentInput.selectionEnd = start;
    }
    
    currentInput.focus();
}

function showKeyboard() {
    document.getElementById('virtual-keyboard').style.display = 'block';
}

function hideKeyboard() {
    document.getElementById('virtual-keyboard').style.display = 'none';
}

function shutdownSystem() {
    if (confirm("Are you sure you want to shutdown the system?")) {
        fetch('/system/shutdown', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                showStatus('info', data.message || 'Shutting down...');
            });
    }
}

function restartSystem() {
    if (confirm("Are you sure you want to restart the system?")) {
        fetch('/system/restart', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                showStatus('info', data.message || 'Restarting...');
            });
    }
}