//This holds a reference to the paragraph tag which will contain status updates.
const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
// The model you will use expects an input image of size 224x224 pixels.
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
// This stores a state value that tells you when a user has stopped clicking the button used to gather data.
const STOP_DATA_GATHER = -1;
// This is a simple array lookup that holds the human-readable names for the possible class predictions.
const CLASS_NAMES = [];

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

let dataCollectorButtons = document.querySelectorAll('button.dataCollector');

for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

// to store the loaded MobileNet model
let mobilenet = undefined;
// When a ‘dataCollector’ button is pressed,
// this will change to be the 1-hot id of the button instead as defined in the HTML code
// The value is initially set to ‘STOP_DATA_GATHER’
// so that your data gather loop (that you write later on) will not gather any data when no button is being pressed.
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
// These store the gathered training data values as you click the ‘dataCollector’ buttons.
let trainingDataInputs = [];
let trainingDataOutputs = [];
// This will keep track of the number of examples sampled for each class.
let examplesCount = [];
let predict = false;
// you will use to store the chopped-up MobileNet model later on
let mobileNetBase = undefined;

// Sometimes it can be useful to define your own printing functions instead of using the console and you will see this function in use later.
function customPrint(line) {
  let p = document.createElement('p');
  p.innerText = line;
  document.body.appendChild(p);
}

/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadMobileNetFeatureModel() {
  const URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json';
  mobilenet = await tf.loadLayersModel(URL);
  STATUS.innerText = 'MobileNet v2 loaded successfully!';
  mobilenet.summary(null, null, customPrint);
  // Scroll down to the end of the output to find the name of the penultimate layer before the classification layers begin.
  // Observe that the name of this layer is ‘global_average_pooling2d_1’.
  // You will understand how this name is used in the subsequent code and description.
  const layer = mobilenet.getLayer('global_average_pooling2d_1');
  // Create a truncated version of the model by calling ‘tf.model’ and
  // pass an object to this function that specifies the inputs’ starting point and
  // the outputs’ ending point. Note that the inputs are the same as the original inputs,
  // so use ‘mobilenet.inputs’ to refer to those and for the outputs you can use ‘lastLayer.outputs’.
  // Store the result of this chopped up model in a ‘mobileNetBase’ variable.
  mobileNetBase = tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  mobileNetBase.summary();

  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobileNetBase.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log("output shape: ")//[1,1280]
    console.log(answer.shape)
  });
}

loadMobileNetFeatureModel();

let model = tf.sequential();
// An input shape of 1280 (which you found was the size of the output of mobilenet)
model.add(tf.layers.dense({inputShape: [1280], units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));
model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  // Adam changes the learning rate over time which is useful.
  optimizer: 'adam',
  // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
  // Else categoricalCrossentropy is used if more than 2 classes.
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy',
  // As this is a classification problem you can record accuracy in the logs too!
  metrics: ['accuracy']
});

// Create a function to check if the browser supports ‘getUserMedia’ and allows access to the webcam.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  console.log("Enable camera")
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640,
      height: 480
    };
    // Activate the webcam stream.
    // Call ‘navigator.mediaDevices.getUserMedia’ with the constraints defined in the previous step and wait for the stream to be returned.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      // Set the video element to play the returned stream.
      VIDEO.srcObject = stream;
      // Add an event listener on the video element for the stream ‘loadeddata’ event.
      // Remember to set the ‘videoPlaying’ variable value to ‘true’ once the stream loads.
      VIDEO.addEventListener('loadeddata', function() {
        videoPlaying = true;
        // Once the video is playing successfully, remove the ENABLE_CAM_BUTTON from the view to prevent it from being clicked again.
        // This is done by setting its class to ‘removed’.
        ENABLE_CAM_BUTTON.classList.add('removed');
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}

/**
 * Handle Data Gather for button mouseup/mousedown.
 **/
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute('data-1hot'));
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

/**
 * This function is responsible for sampling images from the webcam video, passing them through the mobilenet model,
 **/
function dataGatherLoop() {
  // First check if the program is in a state where data should be gat
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    let imageFeatures = tf.tidy(function() {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      // Note that this will stretch the image as your webcam image is 640x480 pixels in size,
      // where as ImageNet needs a square-shaped image that is 224x224 pixels in size.
      // You may choose to try and crop a square from this image for even better results
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT,MOBILE_NET_INPUT_WIDTH], true); //Also set ‘align corners’ to ‘true’
      // The next step is to normalize the image data.
      // Since you have used ‘tf.browser.fromPixels’ the image data is always between the range 0 and 255.
      // To normalize the data, you can simply divide the ‘resizedTensorFrame’ by 255
      // to ensure all resulting values lie between 0 and 1.
      let normalizedTensorFrame = resizedTensorFrame.div(255);
      // You now call ‘mobilenet.predict’ and pass the expanded version of the ‘normalizedTensorFrame’
      // (use expandDims to convert it to a tensor2d to account for the batch of 1).
      // You can then immediately squeeze the result to squash it back to a tensor1d
      // which is returned and assigned to the ‘imageFeatures’ variable (which captures the results from the ‘tf.tidy’ function).
      return mobileNetBase.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }

    examplesCount[gatherDataState]++;
    STATUS.innerText = '';

    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
    }
    // call ‘window.requestAnimationFrame’ with ‘dataGatherLoop’ passed as a parameter,
    // to recursively call this function all over again.
    // This will continue to sample frames from the video until the button mouseup is detected,
    // and gatherDataState becomes STOP_DATA_GATHER at which point this data gather loop will end
    window.requestAnimationFrame(dataGatherLoop);
  }
}

async function trainAndPredict() {
  // Stop any current predictions from taking place by setting ‘predict’ to ‘false’.
  predict = false;
  // Shuffle your input and output arrays
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  // Convert your output array to be a tensor1d with type ‘int32’ so it is ready to be used in a 1-hot encoding
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  // Use the ‘tf.oneHot function’ with this ‘outputsAsTensor’ variable and the max number of classes to encode
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  // ‘trainingDataInputs’ is currently an array of tensors. In order to use these for training,
  // you will need to convert this into a regular tensor2d. To do this, use a function from the TensorFlow.js
  // library called ‘tf.stack’, which takes an array of tensors and stacks them together to produce
  // a single tensor as output. In this case, a tensor2d is returned, which is what you need for training
  // - a batch of 1-dimensional inputs that are each 1,024 in length and contain the features recorded.
  let inputsAsTensor = tf.stack(trainingDataInputs);

  let results = await model.fit(inputsAsTensor, oneHotOutputs,
    {
      shuffle: true,
      batchSize: 5,
      epochs: 5,
      callbacks: {onEpochEnd: logProgress}
    });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  // Set ‘predict’ back to true to allow predictions to take place again
  predict = true;

  // Make combined model for download.
  let combinedModel = tf.sequential();
  combinedModel.add(mobileNetBase);
  combinedModel.add(model);
  combinedModel.compile({
    optimizer: 'adam',
    loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy'
  });
  combinedModel.summary();
  // Once trained, a new combined model will be created and offered for download along with the weights files,
  // that you can save to your hard drive.
  await combinedModel.save('downloads://my-model');

  // Call the ‘predictLoop’ function to start predicting live webcam images.
  predictLoop();
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

function predictLoop() {
  if (predict) {
    tf.tidy(function() {
      // Get the image features for the current image by grabbing a frame from the webcam using ‘tf.browser.fromPixels’,
      // then normalize it by dividing its values by 255
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);

      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT,MOBILE_NET_INPUT_WIDTH], true);


      let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());

      let prediction = model.predict(imageFeatures).squeeze();

      // With this ‘prediction’, you can find the index that has the highest value using the ‘argMax’ function
      // and then convert this resulting tensor to an array sequentially using ‘arraySync’.
      // This value is stored in a variable called ‘highestIndex’.
      let highestIndex = prediction.argMax().arraySync();

      // You can also get the actual prediction confidence scores in the same manner by calling ‘arraySync’
      // on the ‘prediction’ tensor directly and store the result in a variable named  ‘predictionArray’.
      let predictionArray = prediction.arraySync();

      STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
    });

    // Once ready, use the ‘window.requestAnimationFrame’ to call the ‘predictionLoop’ all over again.
    // This allows you to get real time classification on your video stream.
    // It will continue until ‘predict’ is set to false (which will be set to false if you choose to train a new model with new data).
    window.requestAnimationFrame(predictLoop);

  }

}

// Stop any currently running prediction loops by setting ‘predict’ to false.
//
// Delete all content in the ‘examplesCount' array by using the ‘splice(0)’ method you learned in prior chapters.
//
// Go through all the currently recorded ‘trainingDataInputs’ and ensure that you dispose of each tensor contained within it in order to free up memory again.
//
// Next Call ‘splice(0)’ on both the ‘trainingDataInputs’ and ‘trainingDataOutputs’ arrays to clear those too.
// Note: If you had called ‘splice(0)’ on the ‘trainingDataInputs’ array before disposing of the tensors contained within,
// the tensors would be unreachable but still in memory and not disposed of, which could cause a memory leak.
//
// Set the status text to something appropriate and print out the tensors left in memory as a sanity check.
// Remember that since the MobileNet model and the multi-layer perceptron are not yet disposed of,
// a few hundred tensors will still be left in the device memory which is expected.
// This allows you to reuse them with new training data if you decide to train the model again after a reset.
function reset() {

  predict = false;

  examplesCount.splice(0);

  for (let i = 0; i < trainingDataInputs.length; i++) {

    trainingDataInputs[i].dispose();

  }

  trainingDataInputs.splice(0);

  trainingDataOutputs.splice(0);

  STATUS.innerText = 'No data collected';



  console.log('Tensors in memory: ' + tf.memory().numTensors);

}



//Fine-Tuning the Base Model

//In this project, you trained the prediction head from the image features generated by the truncated MobileNet model.
// This keeps the truncated MobileNet model separate when training,
// so its weights are not changed when you train the new prediction head.

// However, sometimes it is desirable to carefully calibrate the values of the last few layers
// of the base model after a prediction head has been trained. This is known as fine-tuning.

// In this case, another approach is to force all layers in the base model to not be trainable
// by setting their trainable property to false as shown below.
// At the same time, you can grab a reference to the layers that you want to fine-tune later on
// by checking their names and pushing the ones that match onto an array to use later.

/*
let fineTuningLayers = [];
for (const layer of mobileNetBase.layers) {
  layer.trainable = false;
  if (layer.name.indexOf('layer_name_of_interest') === 0) {
    fineTuningLayers.push(layer);
  }
}
*/
// You can now combine the new prediction head with the truncated base model as you did before,
// and train like normal, as only the prediction head will be trainable at this point.

// Once trained you can then go back to one of the fine-tuning layers and set its trainable property
// to true and then perform further training such that it also updates in the training process too,
// allowing you to fine-tune the model to potentially get slightly better results.

/*
fineTuningLayers[fineTuneIndex].trainable = true;

trainAgain();
*/


