// ===============================
// API Configuration
// ===============================
const API_URL = 'https://stock-market-api.onrender.com/predict';


// ===============================
// LOAD COMPANIES FROM DATASET
// ===============================
async function loadCompanies() {

    try {

        const response = await fetch("http://127.0.0.1:8000/companies");

        const companies = await response.json();

        const select = document.getElementById("company");

        companies.forEach(company => {

            const option = document.createElement("option");

            option.value = company;
            option.textContent = company;

            select.appendChild(option);

        });

    } catch (error) {

        console.error("Error loading companies:", error);

    }

}


// ===============================
// LOAD LATEST PRICE WHEN COMPANY SELECTED
// ===============================
async function loadLatestPrice(company) {

    if (!company) return;

    try {

        const response = await fetch(
            `http://127.0.0.1:8000/latest/${company}`
        );

        const data = await response.json();

        document.getElementById("open").value = data.open;
        document.getElementById("high").value = data.high;
        document.getElementById("low").value = data.low;
        document.getElementById("close").value = data.close;

    } catch (error) {

        console.error("Error loading latest price:", error);

    }

}


// ===============================
// DOM Elements
// ===============================
const predictionForm = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const resultSection = document.getElementById('resultSection');
const currentPrice = document.getElementById('currentPrice');
const predictedPrice = document.getElementById('predictedPrice');
const trendSection = document.getElementById('trendSection');
const trendIcon = document.getElementById('trendIcon');
const trendValue = document.getElementById('trendValue');

// Form Elements
const companySelect = document.getElementById('company');
const openInput = document.getElementById('open');
const highInput = document.getElementById('high');
const lowInput = document.getElementById('low');
const closeInput = document.getElementById('close');


// ===============================
// Hide error message
// ===============================
function hideError() {

    if (errorMessage)
        errorMessage.style.display = 'none';

}


// ===============================
// Show error message
// ===============================
function showError(message) {

    if (errorText)
        errorText.textContent = message;

    if (errorMessage)
        errorMessage.style.display = 'flex';

    if (resultSection)
        resultSection.style.display = 'none';

}


// ===============================
// Show loading state
// ===============================
function showLoading() {

    if (predictBtn)
        predictBtn.disabled = true;

    if (btnText)
        btnText.style.display = 'none';

    if (btnLoader)
        btnLoader.style.display = 'inline-block';

    hideError();

}


// ===============================
// Hide loading state
// ===============================
function hideLoading() {

    if (predictBtn)
        predictBtn.disabled = false;

    if (btnText)
        btnText.style.display = 'inline';

    if (btnLoader)
        btnLoader.style.display = 'none';

}


// ===============================
// Validate form inputs
// ===============================
function validateInputs(data) {

    const { company, open, high, low, close } = data;

    if (!company)
        throw new Error('Please select a company');

    if (open <= 0 || high <= 0 || low <= 0 || close <= 0)
        throw new Error('All prices must be greater than 0');

    if (high < low)
        throw new Error('High price cannot be less than low price');

    if (high < open || high < close)
        throw new Error('High price must be the highest value');

    if (low > open || low > close)
        throw new Error('Low price must be the lowest value');

    return true;

}


// ===============================
// Format currency
// ===============================
function formatCurrency(value) {

    return `₹${parseFloat(value).toFixed(2)}`;

}


// ===============================
// Display prediction results
// ===============================
function displayResults(data, closePrice) {

    const prediction = data.prediction;
    const trend = data.trend;

    if (resultSection)
        resultSection.style.display = 'block';

    if (currentPrice)
        currentPrice.textContent = formatCurrency(closePrice);

    if (predictedPrice)
        predictedPrice.textContent = formatCurrency(prediction);

    const trendCard = document.querySelector('.trend-card');

    if (trend === 'UP') {

        trendIcon.textContent = '📈';
        trendValue.textContent = 'UPWARD';
        trendValue.className = 'trend-value up';

        if (trendCard)
            trendCard.className = 'trend-card up';

    } else {

        trendIcon.textContent = '📉';
        trendValue.textContent = 'DOWNWARD';
        trendValue.className = 'trend-value down';

        if (trendCard)
            trendCard.className = 'trend-card down';

    }

}


// ===============================
// Make prediction API call
// ===============================
async function makePrediction(requestData) {

    try {

        const response = await fetch(API_URL, {

            method: 'POST',

            headers: {
                'Content-Type': 'application/json',
            },

            body: JSON.stringify(requestData),

        });

        if (!response.ok)
            throw new Error("Server error");

        return await response.json();

    } catch (error) {

        throw new Error(
            'Cannot connect to backend server.'
        );

    }

}


// ===============================
// Handle form submission
// ===============================
async function handleSubmit(event) {

    event.preventDefault();

    const formData = {

        company: companySelect.value.trim(),
        open: parseFloat(openInput.value),
        high: parseFloat(highInput.value),
        low: parseFloat(lowInput.value),
        close: parseFloat(closeInput.value),

    };

    try {

        validateInputs(formData);

        showLoading();

        const result = await makePrediction(formData);

        displayResults(result, formData.close);

    } catch (error) {

        showError(error.message);

    } finally {

        hideLoading();

    }

}


// ===============================
// Add input validation listeners
// ===============================
function addInputValidation() {

    const numberInputs = [
        openInput,
        highInput,
        lowInput,
        closeInput
    ];

    numberInputs.forEach(input => {

        if (input)
            input.addEventListener('input', hideError);

    });

    if (companySelect)
        companySelect.addEventListener('change', hideError);

}


// ===============================
// Initialize the application
// ===============================
function init() {

    loadCompanies();

    if (predictionForm)
        predictionForm.addEventListener(
            'submit',
            handleSubmit
        );

    addInputValidation();

    console.log(
        'Stock Market Prediction App initialized'
    );

}


// ===============================
// Company change listener
// ===============================
document.addEventListener(
    "DOMContentLoaded",
    function () {

        const companySelect =
            document.getElementById("company");

        if (companySelect) {

            companySelect.addEventListener(
                "change",
                function () {

                    loadLatestPrice(this.value);

                });

        }

    }
);


// ===============================
// Run initialization
// ===============================
if (document.readyState === 'loading') {

    document.addEventListener(
        'DOMContentLoaded',
        init
    );

} else {

    init();

}
