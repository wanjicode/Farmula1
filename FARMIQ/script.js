// --- CONFIGURATION ---
const API_BASE_URL = 'http://127.0.0.1:5000'; // Your Flask backend URL

// --- DOM ELEMENTS ---
const form = document.getElementById('predictionForm');
const areaSelect = document.getElementById('area');
const itemSelect = document.getElementById('item');
const predictBtn = document.getElementById('predictBtn');
const loadingDiv = document.getElementById('loading');
const resultContainer = document.getElementById('resultContainer');

// --- FUNCTIONS ---

/**
 * Fetch initial options (areas and crops) from the backend
 */
async function populateSelectOptions() {
    try {
        const response = await fetch(`${API_BASE_URL}/get_options`);
        if (!response.ok) throw new Error('Network response was not ok.');

        const data = await response.json();

        if (data.success) {
            // Populate areas
            areaSelect.innerHTML = '<option value="">Select an area</option>';
            data.areas.forEach(area => {
                const option = new Option(area, area);
                areaSelect.add(option);
            });

            // Populate crops
            itemSelect.innerHTML = '<option value="">Select a crop</option>';
            data.crops.forEach(crop => {
                const option = new Option(crop, crop);
                itemSelect.add(option);
            });
        } else {
            throw new Error('Failed to fetch options from server.');
        }
    } catch (error) {
        console.error('Error fetching options:', error);
        areaSelect.innerHTML = '<option value="">Error loading</option>';
        itemSelect.innerHTML = '<option value="">Error loading</option>';
    }
}

/**
 * Display the final prediction results and charts
 * @param {object} data - The response data from the /predict endpoint
 */
function displayResults(data) {
    resultContainer.innerHTML = `
        <div class="result-header">
            <h2>Prediction for ${data.inputs.item} in ${data.inputs.area}</h2>
        </div>
        
        <div class="prediction-display">
            <div class="value">${data.predictions.kg_per_ha.toLocaleString('en-US', { maximumFractionDigits: 0 })}</div>
            <div class="unit">Predicted Kilograms per Hectare (kg/ha)</div>
        </div>

        <div class="charts-section">
            <h3>Yield Analysis Charts</h3>
            <div class="chart-grid">
                <div class="chart-item">
                    <h4 class="chart-title">💧 Rainfall Impact</h4>
                    <img src="data:image/png;base64,${data.charts.rainfall_chart}" class="chart-image" alt="Rainfall Impact Chart">
                </div>
                <div class="chart-item">
                    <h4 class="chart-title">🌡️ Temperature Impact</h4>
                    <img src="data:image/png;base64,${data.charts.temperature_chart}" class="chart-image" alt="Temperature Impact Chart">
                </div>
                 <div class="chart-item">
                    <h4 class="chart-title">🌱 Crop Comparison</h4>
                    <img src="data:image/png;base64,${data.charts.crop_comparison_chart}" class="chart-image" alt="Crop Comparison Chart">
                </div>
            </div>
        </div>
    `;
    resultContainer.classList.add('show');
}

/**
 * Display an error message
 * @param {string} message - The error message to show
 */
function displayError(message) {
    resultContainer.innerHTML = `<div class="error-message">❌ ${message}</div>`;
    resultContainer.classList.add('show');
}

/**
 * Handle the form submission event
 * @param {Event} e - The form submission event
 */
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // UI updates
    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing...';
    loadingDiv.style.display = 'block';
    resultContainer.style.display = 'none';

    // Collect form data
    const formData = {
        area: document.getElementById('area').value,
        item: document.getElementById('item').value,
        year: parseInt(document.getElementById('year').value),
        rainfall: parseFloat(document.getElementById('rainfall').value),
        pesticides: parseFloat(document.getElementById('pesticides').value),
        temperature: parseFloat(document.getElementById('temperature').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error(`Server responded with status: ${response.status}`);
        
        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            displayError(data.error || 'An unknown prediction error occurred.');
        }

    } catch (error) {
        console.error('Prediction request failed:', error);
        displayError('Could not connect to the backend server. Please ensure it is running.');
    } finally {
        // Restore UI
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict Yield';
        loadingDiv.style.display = 'none';
        resultContainer.style.display = 'block';
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }
}

// --- EVENT LISTENERS ---

// Populate dropdowns when the page loads
document.addEventListener('DOMContentLoaded', populateSelectOptions);

// Listen for form submission
form.addEventListener('submit', handleFormSubmit);