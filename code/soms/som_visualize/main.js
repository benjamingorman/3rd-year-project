class SOM {
    constructor(rows, cols, input_dims, data) {
        this.rows = rows;
        this.cols = cols;
        this.input_dims = input_dims;
        this.data = data;
    }
}

let CURRENT_SOM;

$(document).ready(function() {
    visualizeSOM(loadDefaultSOM());
});

$("#viz_box_inner").panzoom({
    $zoomIn: $(".zoom-in"),
    $zoomOut: $(".zoom-out"),
    $zoomRange: $(".zoom-range"),
    $reset: $(".reset")
});

$("#viz_box").on('mousewheel.focal', function(e) {
    e.preventDefault();
    let delta = e.delta || e.originalEvent.wheelDelta;
    let zoomOut = delta ? delta < 0 : e.originalEvent.deltaY > 0;
    $("#viz_box_inner").panzoom('zoom', zoomOut, {
        animate: false,
        focal: e
    });
});

$("#som_description_text").change(function() {
    console.log("som description changed");
    let som = loadSOMFromDescription();
    visualizeSOM(som);
});

$("#som_file_select").change(function(evt) {
    console.log(evt.target.files);
    if (evt.target.files.length > 0) {
        let file = evt.target.files[0];
        let reader = new FileReader();
        reader.onload = function() {
            $("#som_description_text").text(reader.result);
            $("#som_description_text").trigger("change");
        }
        reader.readAsText(file);
    }
});

$("#input_pattern_test").click(function() {
    let pattern = $("#input_pattern_text").val();
    let parts = pattern.split(",");

    if (parts.length != CURRENT_SOM.input_dims) {
        console.log("Couldn't parse input pattern.", parts);
    }
    else {
        let input_vec = parts.map(parseFloat);
        console.log("Input vec", input_vec);
        
        let num_neurons = CURRENT_SOM.rows * CURRENT_SOM.cols; 
        let bmu = 0;
        let best_dist = Number.MAX_VALUE;
        let worst_dist = Number.MIN_VALUE;

        for (let i=0; i < num_neurons; ++i) {
            let weight_vec = getNeuronWeightVector(CURRENT_SOM, i);
            let dist = vector_magnitude(vector_difference(input_vec, weight_vec));

            if (dist < best_dist) {
                bmu = i;
                best_dist = dist;
            }

            if (dist > worst_dist) {
                worst_dist = dist;
            }
        }

        for (let i=0; i < num_neurons; ++i) {
            let weight_vec = getNeuronWeightVector(CURRENT_SOM, i);
            let dist = vector_magnitude(vector_difference(input_vec, weight_vec));
            
            // scale between 0 (best) to 1 (worst)
            let scaled_dist = (dist - best_dist) / (worst_dist - best_dist);
            let match_value = 1 - scaled_dist;

            $("#neuron"+i).css("background", "blue");
            $("#neuron"+i).css("opacity", match_value);
        }
    }
});

function onClickNeuronSquare(evt) {
    let index = $(this).attr("data-index");
    let row = getNeuronRow(CURRENT_SOM, index);
    let col = getNeuronCol(CURRENT_SOM, index);
    let weights = getNeuronWeightVector(CURRENT_SOM, index);
    let weights_str = weights.join("<br>");
    console.log(index);

    $("#neuron_info_index").text(index);
    $("#neuron_info_pos").text(`(${row}, ${col})`);
    $("#neuron_info_weights").html(weights_str);
}

function vector_difference(a, b) {
    let result = [];
    for (let i=0; i < a.length; ++i) {
        result.push(b[i] - a[i]);
    }
    return result;
}

function vector_magnitude(a) {
    let sum_of_squares = 0;
    for (let item of a) {
        sum_of_squares += Math.pow(item, 2);
    }
    return Math.sqrt(sum_of_squares);
}

function loadDefaultSOM() {
    return new SOM(DEFAULT_SOM_ROWS, DEFAULT_SOM_COLS, DEFAULT_SOM_INPUT_DIMS, DEFAULT_SOM_DATA);
}

function loadSOMFromDescription() {
    let desc = $("#som_description_text").text();
    //console.log(desc);

    let lines = desc.split('\n');
    let [rows_str, cols_str, input_dims_str] = lines[0].split(',');
    let rows = parseInt(rows_str);
    let cols = parseInt(cols_str);
    let input_dims = parseInt(input_dims_str);
    
    let data = [];

    for (let i=1; i < rows * cols + 1; ++i) {
        let neuron_data = [];
        for (let item of lines[i].split(",")) {
            if (item.length)
                neuron_data.push(parseFloat(item));
        }

        if (neuron_data.length != input_dims) {
            console.log("Not enough items in neuron data", neuron_data);
            debugger;
        }
        data.push(neuron_data);
    }

    if (data.length != rows * cols) {
        console.log("Not enough items in data", data);
        debugger;
    }

    let som = new SOM(rows, cols, input_dims, data);
    console.log(som);
    return som;
}

function getNeuronWeightVector(som, neuron) {
    return som.data[neuron];
}

function getNeuronRow(som, i) {
    return Math.floor(i / som.cols);
}

function getNeuronCol(som, i) {
    return i % som.cols;
}

function visualizeSOM(som) {
    console.log("Visualizing som...");
    CURRENT_SOM = som;
    let box = $("#viz_box_inner");
    box.empty();
    box.panzoom("reset");

    let num_neurons = som.rows * som.cols;
    let origin_x = ($("#viz_box_inner").width() - som.cols*20) / 2.0;
    let origin_y = ($("#viz_box_inner").height() - som.rows*20) / 2.0;

    for (let i=0; i < num_neurons; ++i) {
        console.log("Neuron", i);
        let weight_vec = getNeuronWeightVector(som, i);
        let row = getNeuronRow(som, i);
        let col = getNeuronCol(som, i);

        box.append(`<div id='neuron${i}' class='neuron_square' data-index=${i}></div>`);
        let neuron_elem = $("#neuron"+i);
        neuron_elem.click(onClickNeuronSquare);
        let width = neuron_elem.width();
        neuron_elem.css('left', (origin_x + col*width)+'px');
        neuron_elem.css('top',  (origin_y + row*width)+'px');
    }
}
