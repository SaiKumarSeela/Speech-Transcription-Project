const uploadBtn = document.getElementById('upload-btn');
const fileUpload = document.getElementById('file-upload');
const transcriptionDiv = document.getElementById('transcription');
const tabContent = document.getElementById('tab-content');
const loader = document.getElementById('loader');
const statusUpdates = document.getElementById('status-updates');

uploadBtn.addEventListener('click', async() => {
    const file = fileUpload.files[0];
    if (!file) {
        alert('Please select a file first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        loader.style.display = 'block';
        uploadBtn.disabled = true;
        statusUpdates.innerHTML = '';

        // Display success message for file upload
        statusUpdates.innerHTML += '<p>File uploaded successfully. Processing started...</p>';

        const eventSource = new EventSource('/transcribe/');

        eventSource.onmessage = function(event) {
            const data = event.data;
            if (data.startsWith('status: ')) {
                statusUpdates.innerHTML += `<p>${data.substring(8)}</p>`;
            }
            if (data === 'status: Processing complete!') {
                eventSource.close();
                getTranscription();
            }
        };

        eventSource.onerror = function(event) {
            console.error('EventSource failed:', event);
            eventSource.close();
            loader.style.display = 'none';
            uploadBtn.disabled = false;
            statusUpdates.innerHTML += '<p>Error: Failed to process the audio file. Please try again or contact support if the issue persists.</p>';
        };

    } catch (error) {
        console.error('Error:', error);
        statusUpdates.innerHTML += '<p>Error: Failed to upload or process the audio file. Please try again or contact support if the issue persists.</p>';
        loader.style.display = 'none';
        uploadBtn.disabled = false;
    }
});

async function getTranscription() {
    try {
        const response = await fetch('/transcription/');
        const data = await response.json();
        if (data.conversation) {
            displayTranscription(data.conversation);
        } else {
            statusUpdates.innerHTML += '<p>Error: Failed to fetch transcription. Please try again.</p>';
        }
    } catch (error) {
        console.error('Error fetching transcription:', error);
        statusUpdates.innerHTML += '<p>Error: Failed to fetch transcription. Please try again or contact support if the issue persists.</p>';
    } finally {
        loader.style.display = 'none';
        uploadBtn.disabled = false;
    }
}

function displayTranscription(conversation) {
    transcriptionDiv.innerHTML = conversation.map(entry =>
        `<div class="transcript-line">${entry}</div>`
    ).join('');
}

async function showTab(tabName) {
    let content = '';
    try {
        if (tabName === 'summary') {
            const response = await fetch('/summary/');
            const data = await response.json();
            content = displaySummary(data);
        } else if (tabName === 'stats') {
            const response = await fetch('/stats/');
            const data = await response.json();
            content = displayStats(data);
        }
        tabContent.innerHTML = content;
    } catch (error) {
        console.error(`Error fetching ${tabName}:`, error);
        tabContent.innerHTML = `<p>Failed to fetch ${tabName}. Please try again or contact support if the issue persists.</p>`;
    }
}

function displaySummary(summaryData) {
    if (summaryData.error) {
        return `<p>${summaryData.error}</p>`;
    }
    return `
        <table>
            <tr>
                <th>Speaker</th>
                <th>Summary</th>
            </tr>
            ${summaryData.Speaker.map((speaker, index) => `
                <tr>
                    <td>${speaker}</td>
                    <td>${summaryData.Summary[index]}</td>
                </tr>
            `).join('')}
        </table>
    `;
}

function displayStats(data) {
    if (data.error) {
        return `<p>${data.error}</p>`;
    }
    const speakerStats = Object.entries(data.words_by_speaker)
        .map(([speaker, count]) => `<tr><td>Words by ${speaker}</td><td>${count}</td></tr>`)
        .join('');

    return `
        <table>
            <tr><td>Audio Duration (m)</td><td>${data.audio_duration.toFixed(2)}</td></tr>
            <tr><td>Total Words</td><td>${data.total_words}</td></tr>
            ${speakerStats}
        </table>
    `;
}