// Function to preview the image
function previewImage() {
    const fileInput = document.getElementById("imageInput");
    const previewContainer = document.getElementById("imagePreviewContainer");
    const previewImage = document.getElementById("imagePreview");

    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
        }

        reader.readAsDataURL(file);
    }
}

// Function to make prediction
async function predict() {
    const fileInput = document.getElementById("imageInput");
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("predictionResult").innerText = "Prediction: " + data.prediction;
}
