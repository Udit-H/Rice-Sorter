// Global variables for batch processing - separate rice and dal results
let riceResults = {
  total_objects: 0,
  full_grain_count: 0,
  broken_grain_count: 0,
  chalky_count: 0,
  black_count: 0,
  yellow_count: 0,
  brown_count: 0,
  stone_count: 0,
  husk_count: 0,
  broken_percentages: {
    '25%': 0,
    '50%': 0,
    '75%': 0
  }
};

let dalResults = {
  total_objects: 0,
  full_grain_count: 0,
  broken_grain_count: 0,
  black_dal: 0,  // Added black_dal field
  broken_percentages: {
    '25%': 0,
    '50%': 0,
    '75%': 0
  }
};

// Track what was last analyzed
let lastAnalyzed = null; // 'rice' or 'dal'

function triggerUpload() {
  document.getElementById('fileInput').click();
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('fileInput').addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      const formData = new FormData();
      formData.append('file', file);
      
      const overlay = document.getElementById('loadingOverlay');
      overlay.style.display = 'flex';
      
      fetch("/upload", {
        method: "POST",
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        if (data.success && data.image_url) {
          document.getElementById('displayImage').src = data.image_url;
          document.getElementById('retakeButton').classList.remove('d-none');
        } else {
          alert("Upload failed: " + (data.error || "Unknown error"));
        }
      })
      .catch(error => {
        console.error("Error uploading file:", error);
        alert("Error uploading file");
      })
      .finally(() => {
        overlay.style.display = 'none';
        // Reset input so same file can be selected again if needed
        document.getElementById('fileInput').value = '';
      });
    }
  });
});

function captureImage() {
  fetch("/capture", { method: "POST" })
    .then(response => response.json())
    .then(data => {
      if (data.image_url) {
        document.getElementById('displayImage').src = data.image_url + "?t=" + new Date().getTime();
        document.getElementById('retakeButton').classList.remove('d-none');
      }
    })
    .catch(error => console.error("Capture failed:", error));
}

function retakeImage() {
  document.getElementById('displayImage').src = "/video_feed";
  document.getElementById('retakeButton').classList.add('d-none');
}

// Helper function to reset rice results
function resetRiceResults() {
  riceResults = {
    total_objects: 0,
    full_grain_count: 0,
    broken_grain_count: 0,
    chalky_count: 0,
    black_count: 0,
    yellow_count: 0,
    brown_count: 0,
    stone_count: 0,
    husk_count: 0,
    broken_percentages: {
      '25%': 0,
      '50%': 0,
      '75%': 0
    }
  };
}

// Helper function to reset dal results
function resetDalResults() {
  dalResults = {
    total_objects: 0,
    full_grain_count: 0,
    broken_grain_count: 0,
    black_dal: 0,  // Added black_dal field
    broken_percentages: {
      '25%': 0,
      '50%': 0,
      '75%': 0
    }
  };
}

function analyzeSelection() {
  // Reset dal results when analyzing rice
  resetDalResults();
  
  const overlay = document.getElementById('loadingOverlay');
  const analyzeBtn = document.getElementById('analyzeBtn');

  overlay.style.display = 'flex';
  analyzeBtn.disabled = true;

  let displayedImage = document.getElementById('displayImage').src;
  let imagePath = new URL(displayedImage, window.location.origin).pathname;

  fetch("/process_image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_path: imagePath })
  })
  .then(response => response.json())
  .then(data => {
    if (data.processed_image_url) {
      document.getElementById('displayImage').src = data.processed_image_url + "?t=" + new Date().getTime();
    }
    
    // Set last analyzed type
    lastAnalyzed = 'rice';
    
    // Accumulate rice results
    riceResults.total_objects += (data.total_objects || 0);
    riceResults.full_grain_count += (data.full_grain_count || 0);
    riceResults.broken_grain_count += (data.broken_grain_count || 0);
    riceResults.chalky_count += (data.chalky_count || 0);
    riceResults.black_count += (data.black_count || 0);
    riceResults.yellow_count += (data.yellow_count || 0);
    riceResults.brown_count += (data.brown_count || 0);
    riceResults.stone_count += (data.stone_count || 0);
    riceResults.husk_count += (data.husk_count || 0);
    
    // Accumulate broken percentages if available
    if (data.broken_percentages) {
      riceResults.broken_percentages['25%'] += (data.broken_percentages['25%'] || 0);
      riceResults.broken_percentages['50%'] += (data.broken_percentages['50%'] || 0);
      riceResults.broken_percentages['75%'] += (data.broken_percentages['75%'] || 0);
    }
    
    // Format broken rice percentages
    let brokenDetails = '';
    if (riceResults.broken_percentages) {
      const bp = riceResults.broken_percentages;
      brokenDetails = `
        <div class="broken-details">
          <div>25% broken: ${bp['25%'] || 0}</div>
          <div>50% broken: ${bp['50%'] || 0}</div>
          <div>75% broken: ${bp['75%'] || 0}</div>
        </div>
      `;
    }
    
    // Update UI with rice results
    document.getElementById('resultsContent').innerHTML = `
      <div class="result-section">
        <div class="result-section-title">Rice Batch Summary</div>
        <div class="result-item">
          <strong>Total Grain Count:</strong>
          <span class="result-value">${riceResults.total_objects}</span>
        </div>
      </div>
      
      <div class="result-section">
        <div class="result-section-title">Rice Classification</div>
        <div class="result-item">
          <strong>Full Grains:</strong>
          <span class="result-value">${riceResults.full_grain_count}</span>
        </div>
        <div class="result-item">
          <strong>Broken Grains:</strong>
          <span class="result-value">${riceResults.broken_grain_count}</span>
        </div>
        ${brokenDetails}
        <div class="result-item">
          <strong>Chalky Grains:</strong>
          <span class="result-value">${riceResults.chalky_count}</span>
        </div>
        <div class="result-item">
          <strong>Black Grains:</strong>
          <span class="result-value">${riceResults.black_count}</span>
        </div>
        <div class="result-item">
          <strong>Yellow Grains:</strong>
          <span class="result-value">${riceResults.yellow_count}</span>
        </div>
        <div class="result-item">
          <strong>Brown Grains:</strong>
          <span class="result-value">${riceResults.brown_count}</span>
        </div>
      </div>
      
      <div class="result-section">
        <div class="result-section-title">Impurities</div>
        <div class="result-item">
          <strong>Stones:</strong>
          <span class="result-value">${riceResults.stone_count}</span>
        </div>
        <div class="result-item">
          <strong>Husks:</strong>
          <span class="result-value">${riceResults.husk_count}</span>
        </div>
      </div>
      
    `;
  })
  .catch(error => {
    console.error("Error processing image:", error);
    document.getElementById('resultsContent').innerHTML = "<p class='text-danger'>Error processing image</p>";
  })
  .finally(() => {
    overlay.style.display = 'none';
    analyzeBtn.disabled = false;
  });
}

function analyzeDal() {
  // Reset rice results when analyzing dal
  resetRiceResults();
  
  const overlay = document.getElementById('loadingOverlay');
  const analyzeDalBtn = document.getElementById('analyzeDalBtn');

  overlay.style.display = 'flex';
  analyzeDalBtn.disabled = true;

  let displayedImage = document.getElementById('displayImage').src;
  let imagePath = new URL(displayedImage, window.location.origin).pathname;

  fetch("/process_dal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_path: imagePath })
  })
  .then(response => response.json())
  .then(data => {
    if (data.processed_image_url) {
      document.getElementById('displayImage').src = data.processed_image_url + "?t=" + new Date().getTime();
    }
    
    // Set last analyzed type
    lastAnalyzed = 'dal';
    
    // Accumulate dal results
    dalResults.total_objects += (data.full_grain_count + data.broken_grain_count || 0);
    dalResults.full_grain_count += (data.full_grain_count || 0);
    dalResults.broken_grain_count += (data.broken_grain_count || 0);
    dalResults.black_dal += (data.black_dal || 0);  // Added black_dal accumulation
    
    // Accumulate broken percentages if available
    if (data.broken_percent) {
      dalResults.broken_percentages['25%'] += (data.broken_percent['25%'] || 0);
      dalResults.broken_percentages['50%'] += (data.broken_percent['50%'] || 0);
      dalResults.broken_percentages['75%'] += (data.broken_percent['75%'] || 0);
    }
    
    // Format broken dal percentages
    let brokenDetails = '';
    if (dalResults.broken_percentages) {
      const bp = dalResults.broken_percentages;
      brokenDetails = `
        <div class="broken-details">
          <div>25% broken: ${bp['25%'] || 0}</div>
          <div>50% broken: ${bp['50%'] || 0}</div>
          <div>75% broken: ${bp['75%'] || 0}</div>
        </div>
      `;
    }
    
    // Update UI with dal results
    document.getElementById('resultsContent').innerHTML = `
      <div class="result-section">
        <div class="result-section-title">Dal Batch Summary</div>
        <div class="result-item">
          <strong>Total Dal Count:</strong>
          <span class="result-value">${dalResults.total_objects}</span>
        </div>
      </div>
      
      <div class="result-section">
        <div class="result-section-title">Dal Classification</div>
        <div class="result-item">
          <strong>Full Grains:</strong>
          <span class="result-value">${dalResults.full_grain_count}</span>
        </div>
        <div class="result-item">
          <strong>Broken Grains:</strong>
          <span class="result-value">${dalResults.broken_grain_count}</span>
        </div>
        ${brokenDetails}
        <div class="result-item">
          <strong>Black Dal:</strong>
          <span class="result-value">${dalResults.black_dal}</span>
        </div>
      </div>
      
    `;
  })
  .catch(error => {
    console.error("Error processing dal image:", error);
    document.getElementById('resultsContent').innerHTML = "<p class='text-danger'>Error processing dal image</p>";
  })
  .finally(() => {
    overlay.style.display = 'none';
    analyzeDalBtn.disabled = false;
  });
}

// Function to check if all values in an object are zero
function allZero(obj) {
  if (typeof obj !== 'object' || obj === null) {
    return obj === 0;
  }
  
  return Object.values(obj).every(val => {
    if (typeof val === 'object' && val !== null) {
      return allZero(val);
    }
    return val === 0;
  });
}


function resetBatch() {
  // Reset both rice and dal accumulated values
  resetRiceResults();
  resetDalResults();
  retakeImage();
  
  lastAnalyzed = null;
  
  // Update UI to show empty state
  document.getElementById('resultsContent').innerHTML = `
    <div class="empty-state">
      <p>No results yet.</p>
    </div>
  `;
}

function saveResults(){
  let results = {};
  
  if (dalResults.total_objects > 0){
    console.log("Saving dal results...");
    console.log(dalResults);
    results = dalResults;
  } else {
    console.log("Saving rice results...");
    console.log(riceResults);
    results = riceResults;
  }
  
  // Send results to server
  fetch("/save_results", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results)
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      alert("Results saved successfully");
    } else {
      alert("Failed to save: " + data.error);
    }
  })
  .catch(error => {
    console.error("Error saving results:", error);
    alert("Error saving results");
  });
}