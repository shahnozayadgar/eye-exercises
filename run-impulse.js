// Load the inferencing WebAssembly module
const Module = require('./edge-impulse-standalone');
const fs = require('fs');
const { SerialPort, ReadlineParser } = require('serialport');

const path = '/dev/tty.usbmodem101';
const baudRate = 115200;

//const correctSequence = ['default', 'left-right', 'left-right', 'up-down', 'up-down', 'circle-right', 'circle-right', 'circle-left', 'circle-left', 'square-right', 'square-right', 'square-left', 'square-left'];
const correctSequence = ['default', 'up-down', 'left-right', 'circle-right', 'square-right',  'square-left'];


//const correctSequence = ['default', 'left-right', 'up-down', 'circle-right', 'circle-left', 'square-right', 'square-left'];
let currentStep = 0; // Tracks the current step in the sequence

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

    // ================== Modified: Handle Incoming Data with Acknowledgment ==================
    parser.on('data', (line) => {
        const trimmedLine = line.trim();

        // Handle Acknowledgments
        if (trimmedLine === 'BUZZER_ACTIVATED') {
            console.log('Pico acknowledged buzzer activation.');
            return;
        }
        if (trimmedLine === 'LED_ACTIVATED') {
            console.log('Pico acknowledged LED activation.');
            return;
        }

        // =====================================================================

        const parts = trimmedLine.split(',');

        const ax = parseFloat(parts[0]);
        const ay = parseFloat(parts[1]);
        const az = parseFloat(parts[2]);

        if (isNaN(ax) || isNaN(ay) || isNaN(az)) {
            console.error('Received non-numeric data:', trimmedLine);
            return;
        }

        // Append the three-axis data to the buffer
        buffer.push(ax, ay, az);
    });
    // ========================================

    parser.on('error', (err) => {
        console.error('Parser error:', err);
    });

    port.on('error', (err) => {
        console.error('Serial port error:', err);
    });

    console.log('Starting data collection and classification every 5 seconds...');

    // ================== Added: Function to Send Commands to Pico ==================
    const sendCommandToPico = (command) => {
        if (port.isOpen) {
            port.write(`${command}\n`, (err) => {
                if (err) {
                    return console.error('Error sending command to Pico:', err.message);
                }
                console.log(`Command sent to Pico: ${command}`);
            });
        } else {
            console.error('Serial port is not open.');
        }
    };

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

            const classifiedDirections = result.results.map(res => res.label);

            // Find the label with the highest value
            let detectedDirection = 'unknown';
            let maxValue = -Infinity;
            for (let res of result.results) {
                if (res.value > maxValue) {
                    maxValue = res.value;
                    detectedDirection = res.label;
                }
            }
            console.log('Detected Direction:', detectedDirection);
            // ================== Added: Validate Detected Direction and Send Command ==================
            if (detectedDirection === correctSequence[currentStep]) {
                console.log(`Correct direction: ${detectedDirection}`);
                sendCommandToPico('BUZZER_ON'); // Send command to Pico

                currentStep += 1; // Move to the next step in the sequence

                // Reset if the entire sequence is completed
                if (currentStep >= correctSequence.length) {
                    console.log('Sequence completed successfully!');
                    currentStep = 0; // Reset for the next round
                }
            } else {
                console.log(`Incorrect direction. Expected: ${correctSequence[currentStep]}, but got: ${detectedDirection}`);
                // Optionally, reset the sequence or handle incorrect directions
                sendCommandToPico('LED_ON'); // Send command to Pico
                //currentStep = 0; // Reset the sequence
            }
        } catch (err) {
            console.error('Classification error:', err);
        }
    };

    // Schedule classification every 5 seconds
    setInterval(classifyData, DURATION_MS);
})();

