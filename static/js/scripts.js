const uploadForm = document.getElementById('uploadForm');
const plotForm = document.getElementById('plotForm');
const resultDiv = document.getElementById('result');

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
    });
    const data = await response.text();
    resultDiv.innerHTML = data;
});

plotForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(plotForm);
    const response = await fetch('/generate_plot', {
        method: 'POST',
        body: formData,
    });
    if (response.ok) {
        const imgBlob = await response.blob();
        const imgURL = URL.createObjectURL(imgBlob);
        const img = document.createElement('img');
        img.src = imgURL;
        resultDiv.innerHTML = '';
        resultDiv.appendChild(img);
    } else {
        const data = await response.json();
        resultDiv.textContent = 'Error: ${data.error}';
    }
});