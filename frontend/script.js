
const BASE_URL = "https://stock-market-api-1vsa.onrender.com";
const API_URL = `${BASE_URL}/predict`;

async function loadCompanies() {

    try {

        const response =
            await fetch(`${BASE_URL}/companies`);

        if (!response.ok)
            throw new Error("Cannot load companies");

        const companies =
            await response.json();

        const select =
            document.getElementById("company");

        select.innerHTML =
            '<option value="">Select Company</option>';

        companies.forEach(company => {

            const option =
                document.createElement("option");

            option.value = company;
            option.textContent = company;

            select.appendChild(option);

        });

    } catch (error) {

        showError(
            "Backend connection failed"
        );

        console.error(error);

    }

}

async function loadLatestPrice(company) {

    if (!company) return;

    try {

        const response =
            await fetch(
                `${BASE_URL}/latest/${company}`
            );

        if (!response.ok)
            throw new Error("Latest failed");

        const data =
            await response.json();

        document.getElementById("open").value =
            data.open;

        document.getElementById("high").value =
            data.high;

        document.getElementById("low").value =
            data.low;

        document.getElementById("close").value =
            data.close;

    } catch (error) {

        showError(
            "Cannot load price data"
        );

        console.error(error);

    }

}

const predictionForm =
    document.getElementById("predictionForm");

const predictBtn =
    document.getElementById("predictBtn");

const btnText =
    document.getElementById("btnText");

const btnLoader =
    document.getElementById("btnLoader");

const errorMessage =
    document.getElementById("errorMessage");

const errorText =
    document.getElementById("errorText");

const resultSection =
    document.getElementById("resultSection");

const currentPrice =
    document.getElementById("currentPrice");

const predictedPrice =
    document.getElementById("predictedPrice");

const trendIcon =
    document.getElementById("trendIcon");

const trendValue =
    document.getElementById("trendValue");

const companySelect =
    document.getElementById("company");

const openInput =
    document.getElementById("open");

const highInput =
    document.getElementById("high");

const lowInput =
    document.getElementById("low");

const closeInput =
    document.getElementById("close");

function showError(message) {

    errorText.textContent =
        message;

    errorMessage.style.display =
        "flex";

    resultSection.style.display =
        "none";

}

function hideError() {

    errorMessage.style.display =
        "none";

}

function showLoading() {

    predictBtn.disabled = true;
    btnText.style.display = "none";
    btnLoader.style.display =
        "inline-block";

}

function hideLoading() {

    predictBtn.disabled = false;
    btnText.style.display = "inline";
    btnLoader.style.display =
        "none";

}

function formatCurrency(value) {

    return "₹" +
        parseFloat(value)
            .toFixed(2);

}

function displayResults(result, closePrice) {

    resultSection.style.display =
        "block";

    currentPrice.textContent =
        formatCurrency(closePrice);

    predictedPrice.textContent =
        formatCurrency(result.prediction);

    if (result.trend === "UP") {

        trendIcon.textContent =
            "📈";

        trendValue.textContent =
            "UPWARD";

        trendValue.className =
            "trend-value up";

    } else {

        trendIcon.textContent =
            "📉";

        trendValue.textContent =
            "DOWNWARD";

        trendValue.className =
            "trend-value down";

    }

}

async function makePrediction(data) {

    const response =
        await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body:
                JSON.stringify(data)

        });

    if (!response.ok)
        throw new Error(
            "Prediction failed"
        );

    return await response.json();

}

async function handleSubmit(e) {

    e.preventDefault();

    try {

        hideError();

        showLoading();

        const data = {

            company:
                companySelect.value,

            open:
                parseFloat(
                    openInput.value
                ),

            high:
                parseFloat(
                    highInput.value
                ),

            low:
                parseFloat(
                    lowInput.value
                ),

            close:
                parseFloat(
                    closeInput.value
                )

        };

        const result =
            await makePrediction(data);

        displayResults(
            result,
            data.close
        );

    } catch (error) {

        showError(
            "Cannot connect to backend"
        );

    } finally {

        hideLoading();

    }

}

function init() {

    loadCompanies();

    predictionForm.addEventListener(
        "submit",
        handleSubmit
    );

    companySelect.addEventListener(
        "change",
        function () {

            loadLatestPrice(
                this.value
            );

        }
    );

}

document.addEventListener(
    "DOMContentLoaded",
    init
);
