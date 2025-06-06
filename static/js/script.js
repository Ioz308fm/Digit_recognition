const cvsIn = document.getElementById("inputimg");
const ctxIn = cvsIn.getContext("2d");
const divOut = document.getElementById("pred");
let svgGraph = null;
let mouselbtn = false;

// initilize
window.onload = () => {
    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.lineWidth = 7;
    ctxIn.lineCap = "round";
    initProbGraph();
    initProbGraph2(); 
}

function initProbGraph() {
    const dummyData = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; // dummy data for initialize graph
    const margin = { top: 10, right: 10, bottom: 10, left: 20 };
    const width = 250;
    const height = 196;

    const yScale = d3.scaleLinear()
        .domain([9, 0])
        .range([height, 0]);

    svgGraph = d3.select("#probGraph")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svgGraph.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(yScale));

    const barHeight = 20
    svgGraph.selectAll("svg")
        .data(dummyData)
        .enter()
        .append("rect")
        .attr("y", (d, i) => (yScale(i) - barHeight / 2))
        .attr("height", barHeight)
        .style("fill", "green")
        .attr("x", 0)
        .attr("width", d => d * 2)
        .call(d3.axisLeft(yScale));
}

let svgGraph2 = null; // объявить глобально где-то в начале файла

function initProbGraph2() {
    const dummyData = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; // заглушка
    const margin = { top: 10, right: 10, bottom: 10, left: 20 };
    const width = 250;
    const height = 196;

    const yScale = d3.scaleLinear()
        .domain([9, 0])
        .range([height, 0]);

    svgGraph2 = d3.select("#probGraph2")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svgGraph2.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(yScale));

    const barHeight = 20;
    svgGraph2.selectAll("rect")
        .data(dummyData)
        .enter()
        .append("rect")
        .attr("y", (d, i) => (yScale(i) - barHeight / 2))
        .attr("height", barHeight)
        .style("fill", "green")
        .attr("x", 0)
        .attr("width", d => d * 2);
}

cvsIn.addEventListener("mousedown", e => {
    if (e.button === 0) {
        const rect = e.target.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        mouselbtn = true;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    }
    else if (e.button === 2) {
        onClear();  // clear by mouse right button
    }
});

cvsIn.addEventListener("mouseup", e => {
    if (e.button === 0) {
        mouselbtn = false;
        onRecognition();
    }
});

cvsIn.addEventListener("mousemove", e => {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (mouselbtn) {
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
    }
});

cvsIn.addEventListener("touchstart", e => {
    // for touch device
    if (e.targetTouches.length === 1) {
        const rect = e.target.getBoundingClientRect();
        const touch = e.targetTouches[0];
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        ctxIn.beginPath();
        ctxIn.moveTo(x, y);
    }
});

cvsIn.addEventListener("touchmove", e => {
    // for touch device
    if (e.targetTouches.length === 1) {
        const rect = e.target.getBoundingClientRect();
        const touch = e.targetTouches[0];
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        ctxIn.lineTo(x, y);
        ctxIn.stroke();
        e.preventDefault();
    }
});

cvsIn.addEventListener("touchend", e => onRecognition());
cvsIn.addEventListener("contextmenu", e => e.preventDefault());

document.getElementById("clearbtn").onclick = onClear;
function onClear() {
    mouselbtn = false;
    ctxIn.fillStyle = "white";
    ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
    ctxIn.fillStyle = "black";
}

// post digit to server for recognition
async function onRecognition() {
    console.time("time");

    cvsIn.toBlob(async blob => {
        const body = new FormData();
        body.append('img', blob, "dummy.png");

        try {
            const res1 = await fetch("./DigitRecognition", { method: "POST", body });
            const res2 = await fetch("./DigitRecognition2", { method: "POST", body });

            const data1 = await res1.json();
            const data2 = await res2.json();

            showResult(data1, data2); // передаем оба результата
        } catch (error) {
            alert("Ошибка: " + error);
        }

        console.timeEnd("time");
    });
}


function showResult(res1, res2) {
    // Первое предсказание
    const pred1 = res1.pred;
    const probs1 = res1.probs[0]; // берем первый (и единственный) массив вероятностей

    document.getElementById("pred").textContent = pred1;
    document.getElementById("prob").innerHTML =
        "Вероятность : " + probs1[pred1].toFixed(2) + "%";

    svgGraph.selectAll("rect")
        .data(probs1)
        .transition()
        .duration(300)
        .style("fill", (d, i) => i === pred1 ? "blue" : "green")
        .attr("width", d => d * 2);

    // Второе предсказание
    const pred2 = res2.pred;
    const probs2 = res2.probs[0]; // аналогично

    document.getElementById("pred2").textContent = pred2;
    document.getElementById("prob2").innerHTML =
        "Вероятность : " + probs2[pred2].toFixed(2) + "%";

    svgGraph2.selectAll("rect")
        .data(probs2)
        .transition()
        .duration(300)
        .style("fill", (d, i) => i === pred2 ? "blue" : "green")
        .attr("width", d => d * 2);
}