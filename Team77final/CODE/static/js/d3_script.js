let svg, width, height;

document.addEventListener("DOMContentLoaded", function() {
    fetch('/get_date_ranges')
    .then(response => response.json())
    .then(data => {
        const select = document.getElementById('dateRangeSelect');
        data.forEach(dateRange => {
            let option = new Option(dateRange, dateRange);
            select.add(option);
        });
        select.addEventListener('change', function() {
            updateGraph(this.value);
        });
        if (data.length > 0) {
            select.value = data[0];
            updateGraph(data[0]);
        }
    });
});

function updateGraph(dateRange) {
fetch(`/graph_data?date_range=${dateRange}`)
    .then(response => response.json())
    .then(data => {
        svg.selectAll("*").remove(); 
        drawGraph(data); 
    })
    .catch(error => console.error('Error:', error));
}

document.addEventListener("DOMContentLoaded", function() {
    width = window.innerWidth;
    height = window.innerHeight;

    svg = d3.select("#visualization").append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .append("g");

    initializeGraph();
});

function linkArc(d) {
    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const dr = Math.sqrt(dx * dx + dy * dy) * 2;
    return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
}

function initializeGraph() {
    fetch('/graph_data').then(function(response) {
        return response.json();
    }).then(function(data) {
        drawGraph(data); 
    });
}

function updateGraph(dateRange) {
    fetch(`/graph_data?date_range=${dateRange}`)
        .then(response => response.json())
        .then(data => {
            svg.selectAll("*").remove();
            drawGraph(data);
        })
        .catch(error => console.error('Error fetching graph data:', error));
}


function drawGraph(data) {
    const nodes = data.nodes;
    const links = data.links;

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(115).strength(0.1))
        .force("charge", d3.forceManyBody().strength(-40))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.selectAll(".link")
        .data(links)
        .enter().append("path")
        .attr("class", "link")
        .attr("stroke", "#999")
        .attr("fill", "none")
        .attr("d", linkArc);

    const node = svg.selectAll(".node")
        .data(nodes)
        .enter().append("circle")
        .attr("r", 35)
        .attr("class", "node")
        .attr("fill", d => "#003057")
        .on("click", nodeClicked);

    const labels = svg.selectAll(".label")
        .data(nodes)
        .enter().append("text")
        .attr("text-anchor", "middle")
        .attr("class", "label")
        .attr("fill", "white")
        .attr("font-size", "14px")
        .text(d => d.label);

    simulation.on("tick", () => {
        link.attr("d", linkArc);
        node.attr("cx", d => d.x).attr("cy", d => d.y);
        labels.attr("x", d => d.x).attr("y", d => d.y + 5);
        });
}

function zoomToNode(nodeData) {
    const dx = 30; 
    const scale = Math.max(1, Math.min(8, 0.9 / Math.max(dx / width, dx / height) * 1.5)); 
    const translate = [width / 2 - scale * nodeData.x, height / 2 - scale * nodeData.y];

    svg.transition()
        .duration(750)
        .attr("transform", `translate(${translate[0]},${translate[1]}) scale(${scale})`);
}

function formatDate(dateStr) {
    return dateStr.substring(0, 4) + '-' + dateStr.substring(4, 6) + '-' + dateStr.substring(6);
    }

function nodeClicked(d) {
    document.getElementById('zoomOut').style.display = 'block';

    const clusterId = d.id.replace(/[^\d]/g, '');
    const dateRange = document.getElementById('dateRangeSelect').value;
    const endDate = dateRange.split('_to_')[1];
    const reformatted = formatDate(endDate)

    svg.selectAll(".label").style("visibility", "hidden");

    fetch(`/get_stocks?cluster_id=${clusterId}&date=${reformatted}`)
        .then(response => response.json())
        .then(stocks => {
            zoomToNode(d); 
            displayStocks(stocks, d);
        });
}

function resetZoom() {
    svg.transition()
        .duration(750)
        .attr("transform", "translate(0,0) scale(1)")
        .on('end', () => {
            svg.selectAll(".stock-point").remove();
            svg.selectAll("g").remove();
            svg.selectAll(".node").style("visibility", "visible");
            svg.selectAll(".link").style("visibility", "visible");
            svg.selectAll(".label").style("visibility", "visible");
        });

    document.getElementById('zoomOut').style.display = 'none'; 
}

document.getElementById("zoomOut").addEventListener("click", resetZoom);

function displayStocks(stocks, nodeData) {
    svg.selectAll(".node").style("visibility", "hidden");
    svg.selectAll(".link").style("visibility", "hidden");
    svg.selectAll(".label").style("visibility", "hidden");
    svg.selectAll(".circle").style("visibility", "hidden");

const stockGroup = svg.append("g")
    .attr("transform", `translate(${nodeData.x}, ${nodeData.y})`);

const stockSimulation = d3.forceSimulation(stocks)
    .force("charge", d3.forceManyBody().strength(10))
    .force("center", d3.forceCenter(0, 0))
    .force("collision", d3.forceCollide().radius(3))
    .velocityDecay(0.7) 
    .on("tick", ticked);


function ticked() {
    const stockPoints = stockGroup.selectAll(".stock-point")
        .data(stocks);
    
    stockPoints.enter().append("circle")
        .attr("class", "stock-point")
        .attr("r", 2)
        .attr("fill", " #B3A369")
        .merge(stockPoints)  
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    
    
    stockPoints.attr("cx", d => d.x)
        .attr("cy", d => d.y);
    
    stockPoints.exit().remove();

    stockGroup.selectAll(".stock-point")
        .append("title")
        .text(d => `Ticker: $${d.ticker}\nResidual Return 1YR: ${d.residual_return.toFixed(4)}\nRaw Return 1YR: ${d.raw_return.toFixed(4)}`);
    } 

}


