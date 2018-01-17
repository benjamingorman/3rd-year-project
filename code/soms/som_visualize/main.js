class SOM {
    constructor(rows, cols, input_dims, data) {
        this.rows = rows;
        this.cols = cols;
        this.input_dims = input_dims;
        this.data = data; // 2d array, one entry for each neuron
    }

    getNumNeurons() {
        return this.rows * this.cols;
    }

    getNeuronWeights(neuron) {
        return this.data[neuron];
    }

    getNeuronRow(i) {
        return Math.floor(i / this.cols);
    }

    getNeuronCol(i) {
        return i % this.cols;
    }

    findBMU(input_vec) {
        let bmu = 0;
        let best_dist = Number.MAX_VALUE;
        let worst_dist = Number.MIN_VALUE;

        for (let n=0; n < this.getNumNeurons(); ++n) {
            let dist = this.getInputDistToNeuron(input_vec, n);

            if (dist < best_dist) {
                bmu = n;
                best_dist = dist;
            }

            if (dist > worst_dist) {
                worst_dist = dist;
            }
        }

        return {bmu: bmu, best_dist: best_dist, worst_dist: worst_dist}
    }

    getInputDistToNeuron(input_vec, neuron) {
        let weight_vec = this.getNeuronWeights(neuron);
        let dist = vector_magnitude(vector_difference(input_vec, weight_vec));
        return dist;
    }

    // 1 is a perfect match and 0 is nowhere near a match
    getInputMatchToNeuron(input_vec, neuron, best_dist, worst_dist) {
        let dist = this.getInputDistToNeuron(input_vec, neuron);
        let match = normalizeValueInRange(worst_dist, dist, best_dist);
        return match;
    }
}

let CURRENT_SOM;
let CURRENT_PATTERN_LIST = [];

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
        
        let {bmu, best_dist, worst_dist} = CURRENT_SOM.findBMU(input_vec);
        for (let i=0; i < CURRENT_SOM.getNumNeurons(); ++i) {
            let match_value = CURRENT_SOM.getInputMatchToNeuron(input_vec, i, best_dist, worst_dist);

            $("#neuron"+i).css("background", "blue");
            $("#neuron"+i).css("opacity", match_value);
        }
    }
});

$("#input_patterns_file_select").change(function(evt) {
    console.log("foo");
    console.log(evt.target.files);
    if (evt.target.files.length > 0) {
        let file = evt.target.files[0];
        let reader = new FileReader();
        reader.onload = function() {
            let classes_included = $("#classes_included_checkbox").is(":checked");
            loadSampleInputPatterns(reader.result, classes_included);
        }
        reader.readAsText(file);
    }
});

$("#label_neurons_with_classes_btn").click(function() {
    resetNeuronStyles();

    let num_neurons = CURRENT_SOM.getNumNeurons();

    let classes = listPatternClasses();
    let class_indices = {};
    for (let i=0; i < classes.length; ++i) {
        class_indices[classes[i]] = i;
    }

    // Average matches for each neuron to each class
    let neuron_total_matches = [];
    for (let i=0; i < num_neurons; ++i) {
        neuron_total_matches.push(new Array(classes.length).fill(0));
    }

    let count = 0;
    for (let pat of CURRENT_PATTERN_LIST) {
        let class_index = class_indices[pat.pattern_class];
        let {bmu, best_dist, worst_dist} = CURRENT_SOM.findBMU(pat.pattern);

        for (let i=0; i < num_neurons; ++i) {
            let match = CURRENT_SOM.getInputMatchToNeuron(pat.pattern, i, best_dist, worst_dist);
            neuron_total_matches[i][class_index] += match;
        }

        count++;
    }

    let minMaxClassTotals = [];
    for (let i=0; i < classes.length; ++i) {
        let minTotal = argmin(totals => totals[i], neuron_total_matches);
        let maxTotal = argmax(totals => totals[i], neuron_total_matches);
        minMaxClassTotals.push({min: minTotal, max: maxTotal});
    }

    for (let i=0; i < num_neurons; ++i) {
        let totals = neuron_total_matches[i];
        let top_matching_class_value = argmax(x => x, totals);
        let top_matching_class_index = indexmax(x => x, totals);
        let top_matching_class = classes[top_matching_class_index];

        let {min, max} = minMaxClassTotals[top_matching_class_index];

        let class_representation = normalizeValueInRange(min, top_matching_class_value, max);

        $("#neuron"+i).css("background", getClassColor(top_matching_class));
        $("#neuron"+i).css("opacity", class_representation);
    }
});

function normalizeValueInRange(start, middle, end) {
    return (middle - start) / (end - start);
}

function argmax(f, xs) {
    let max_arg = Number.MIN_VALUE;

    for (let i=0; i < xs.length; ++i) {
        let x = xs[i]
        let y = f(x);
        if (y > max_arg) {
            max_arg = y;
        }
    }

    return max_arg;
}

function argmin(f, xs) {
    let min_arg = Number.MAX_VALUE;

    for (let i=0; i < xs.length; ++i) {
        let x = xs[i]
        let y = f(x);
        if (y < min_arg) {
            min_arg = y;
        }
    }

    return min_arg;
}

function indexmax(f, xs) {
    let am = argmax(f, xs);
    return xs.findIndex(x => x == am);
}

function indexmin(f, xs) {
    let am = argmin(f, xs);
    return xs.findIndex(x => x == am);
}

function resetNeuronStyles() {
    console.log("Resetting neuron styles...");
    $(".neuron_square").each(function() {
        let left_px = $(this).css('left');
        let top_px = $(this).css('top');
        $(this).attr("style", "");
        $(this).css('left', left_px);
        $(this).css('top', top_px);
    });
}

function loadSampleInputPatterns(text, classes_included) {
    console.log("Loading sample input patterns...");
    console.log("classes included", classes_included);
    CURRENT_PATTERN_LIST = [];

    for (let patternString of text.split("\n")) {
        if (patternString.length == 0)
            continue;

        let pattern_class;
        let parts = patternString.split(",");
        if (classes_included) {
            // class should be the final element
            pattern_class = parts.pop();
        }

        let pattern = parts.map(parseFloat);
        CURRENT_PATTERN_LIST.push({pattern: pattern, pattern_class: pattern_class});
    }

    for (let i=0; i < CURRENT_PATTERN_LIST.length; ++i) {
        let pat = CURRENT_PATTERN_LIST[i];
        let elem = $(`<div class='input_pattern_tile code_font' data-index=${i}></div>`);
        for (let x of pat.pattern) {
            elem.append($(`<div class='input_pattern_tile_cell'>${x}</div>`));
        }
        $("#input_patterns_box").append(elem);
        elem.click(onClickInputPatternTile);

        if (classes_included) {
            let unique_color = getClassColor(pat.pattern_class); 
            elem.css("background", unique_color);
        }
    }
}

function listPatternClasses() {
    // Set of all classes across each pattern
    let set_of_classes = new Set();

    for (let pat of CURRENT_PATTERN_LIST) {
        let cls = pat.pattern_class;
        set_of_classes.add(cls);
    }

    let list = Array.from(set_of_classes);
    list.sort();
    return list;
}

function onClickInputPatternTile(evt) {
    resetNeuronStyles();
    let pattern_index = $(this).attr("data-index");
    let pat = CURRENT_PATTERN_LIST[pattern_index];

    console.log("Clicked:", pat);

    $("#input_pattern_text").val(pat.pattern.join(","));
    $("#input_pattern_test").click();
}

function onClickNeuronSquare(evt) {
    let index = $(this).attr("data-index");
    let row = CURRENT_SOM.getNeuronRow(index);
    let col = CURRENT_SOM.getNeuronCol(index);
    let weights = CURRENT_SOM.getNeuronWeights(index);
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

function visualizeSOM(som) {
    console.log("Visualizing som...");
    CURRENT_SOM = som;
    let box = $("#viz_box_inner");
    box.empty();
    box.panzoom("reset");

    let num_neurons = som.rows * som.cols;
    let origin_x = ($("#viz_box_inner").width() - som.cols*20) / 2.0;
    let origin_y = ($("#viz_box_inner").height() - som.rows*20) / 2.0;

    for (let n=0; n < num_neurons; ++n) {
        //console.log("Neuron", n);
        let weight_vec = CURRENT_SOM.getNeuronWeights(n);
        let row = CURRENT_SOM.getNeuronRow(n);
        let col = CURRENT_SOM.getNeuronCol(n);

        box.append(`<div id='neuron${n}' class='neuron_square' data-index=${n}></div>`);
        let neuron_elem = $("#neuron"+n);
        neuron_elem.click(onClickNeuronSquare);
        let width = neuron_elem.width();
        neuron_elem.css('left', (origin_x + col*width)+'px');
        neuron_elem.css('top',  (origin_y + row*width)+'px');
    }
}

function getClassColor(cls) {
    let class_list = listPatternClasses();
    let class_index = class_list.findIndex(x => x == cls);
    let color = rainbow(class_list.length, class_index, "c0");
    return color;
}

// Credit Adam Cole
function rainbow(numOfSteps, step, alpha) {
    // This function generates vibrant, "evenly spaced" colours (i.e. no clustering). This is ideal for creating easily distinguishable vibrant markers in Google Maps and other apps.
    // Adam Cole, 2011-Sept-14
    // HSV to RBG adapted from: http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
    let r, g, b;
    let h = step / numOfSteps;
    let i = ~~(h * 6);
    let f = h * 6 - i;
    let q = 1 - f;
    switch(i % 6){
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = q; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = q; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        case 5: r = 1; g = 0; b = q; break;
    }
    let c = "#" + ("00" + (~ ~(r * 255)).toString(16)).slice(-2) + ("00" + (~ ~(g * 255)).toString(16)).slice(-2) + ("00" + (~ ~(b * 255)).toString(16)).slice(-2);
    c += alpha;
    return (c);
}
