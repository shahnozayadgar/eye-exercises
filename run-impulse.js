// Load the inferencing WebAssembly module
const Module = require('./edge-impulse-standalone');
const fs = require('fs');
const { SerialPort, ReadlineParser } = require('serialport');

// Set these to match your environment
const path = '/dev/tty.usbmodem101';
const baudRate = 115200;

// Classifier module
let classifierInitialized = false;
Module.onRuntimeInitialized = function() {
    classifierInitialized = true;
};

class EdgeImpulseClassifier {
    _initialized = false;

    init() {
        if (classifierInitialized === true) return Promise.resolve();

        return new Promise((resolve) => {
            Module.onRuntimeInitialized = () => {
                classifierInitialized = true;
                Module.init();
                resolve();
            };
        });
    }

    getProjectInfo() {
        if (!classifierInitialized) throw new Error('Module is not initialized');
        return Module.get_project();
    }

    classify(rawData, debug = false) {
        if (!classifierInitialized) throw new Error('Module is not initialized');

        let props = Module.get_properties();

        const obj = this._arrayToHeap(rawData);
        let ret = Module.run_classifier(obj.buffer.byteOffset, rawData.length, debug);
        Module._free(obj.ptr);

        if (ret.result !== 0) {
            throw new Error('Classification failed (err code: ' + ret.result + ')');
        }

        let jsResult = {
            anomaly: ret.anomaly,
            results: []
        };

        for (let cx = 0; cx < ret.size(); cx++) {
            let c = ret.get(cx);
            if (props.model_type === 'object_detection' || props.model_type === 'constrained_object_detection') {
                jsResult.results.push({ label: c.label, value: c.value, x: c.x, y: c.y, width: c.width, height: c.height });
            }
            else {
                jsResult.results.push({ label: c.label, value: c.value });
            }
            c.delete();
        }

        if (props.has_visual_anomaly_detection) {
            jsResult.visual_ad_max = ret.visual_ad_max;
            jsResult.visual_ad_mean = ret.visual_ad_mean;
            jsResult.visual_ad_grid_cells = [];
            for (let cx = 0; cx < ret.visual_ad_grid_cells_size(); cx++) {
                let c = ret.visual_ad_grid_cells_get(cx);
                jsResult.visual_ad_grid_cells.push({ label: c.label, value: c.value, x: c.x, y: c.y, width: c.width, height: c.height });
                c.delete();
            }
        }

        ret.delete();

        return jsResult;
    }

    classifyContinuous(rawData, enablePerfCal = true) {
        if (!classifierInitialized) throw new Error('Module is not initialized');

        let props = Module.get_properties();

        const obj = this._arrayToHeap(rawData);
        let ret = Module.run_classifier_continuous(obj.buffer.byteOffset, rawData.length, false, enablePerfCal);
        Module._free(obj.ptr);

        if (ret.result !== 0) {
            throw new Error('Classification failed (err code: ' + ret.result + ')');
        }


        let jsResult = {
            anomaly: ret.anomaly,
            visual_ad_max: ret.visual_ad_max,
            visual_ad_mean: ret.visual_ad_mean,
            results: []
        };

        for (let cx = 0; cx < ret.size(); cx++) {
            let c = ret.get(cx);
            if (props.model_type === 'object_detection' || props.model_type === 'constrained_object_detection' || props.model_type === 'visual_anomaly') {
                jsResult.results.push({ label: c.label, value: c.value, x: c.x, y: c.y, width: c.width, height: c.height });
            }
            else {
                jsResult.results.push({ label: c.label, value: c.value });
            }
            c.delete();
        }

        ret.delete();

        return jsResult;
    }

    getProperties() {
        return Module.get_properties();
    }

    _arrayToHeap(data) {
        let typedArray = new Float32Array(data);
        let numBytes = typedArray.length * typedArray.BYTES_PER_ELEMENT;
        let ptr = Module._malloc(numBytes);
        let heapBytes = new Uint8Array(Module.HEAPU8.buffer, ptr, numBytes);
        heapBytes.set(new Uint8Array(typedArray.buffer));
        return { ptr: ptr, buffer: heapBytes };
    }
}

(async () => {
    // Create an instance of the classifier
    let classifier = new EdgeImpulseClassifier();
    
    try {
        // Initialize the classifier
        await classifier.init();
    } catch (err) {
        console.error('Failed to initialize classifier:', err);
        process.exit(1);
    }

    let project;
    try {
        // Get project information
        project = classifier.getProjectInfo();
        console.log('Running inference for', project.owner, '/', project.name, '(version', project.deploy_version + ')');
    } catch (err) {
        console.error('Error getting project info:', err);
        process.exit(1);
    }

    // Define the duration for data collection (in milliseconds)
    const DURATION_MS = 5000; // 5 seconds

    // Buffer to store incoming data
    let buffer = [];

    // Initialize serial port
    const port = new SerialPort({ path, baudRate }, (err) => {
        if (err) {
            return console.error('Error opening serial port:', err);
        }
        // Port opened successfully
    });

    const parser = port.pipe(new ReadlineParser({ delimiter: '\r\n' }));

    // Handle incoming data
    parser.on('data', (line) => {
        const parts = line.trim().split(',');
        if (parts.length !== 3) {
            // Expecting three values: ax, ay, az
            console.error('Invalid data format. Expected three comma-separated values.');
            return;
        }

        const ax = parseFloat(parts[0]);
        const ay = parseFloat(parts[1]);
        const az = parseFloat(parts[2]);

        if (isNaN(ax) || isNaN(ay) || isNaN(az)) {
            console.error('Received non-numeric data:', line);
            return;
        }

        // Append the three-axis data to the buffer
        buffer.push(ax, ay, az);
    });

    parser.on('error', (err) => {
        console.error('Parser error:', err);
    });

    port.on('error', (err) => {
        console.error('Serial port error:', err);
    });

    console.log('Starting data collection and classification every 5 seconds...');

    // Function to perform classification
    const classifyData = () => {
        if (buffer.length === 0) {
            console.log('No data collected in the last 5 seconds.');
            return;
        }

        // Make a copy of the buffer and clear it for the next window
        const dataToClassify = buffer.slice();
        buffer = [];

        try {
            const result = classifier.classify(dataToClassify);
            delete result.anomaly; // Remove anomaly if not needed
            console.log('Prediction:', result);
        } catch (err) {
            console.error('Classification error:', err);
        }
    };

    // Schedule classification every 5 seconds
    setInterval(classifyData, DURATION_MS);
})();