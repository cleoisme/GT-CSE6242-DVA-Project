document.addEventListener("DOMContentLoaded", function() {
    fetch('/data?table=sample_dist_mat') 
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log(data)
    })
    .catch(error => console.error('Error fetching data:', error));
});
