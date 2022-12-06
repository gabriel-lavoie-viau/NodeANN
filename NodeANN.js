const maxApi = require("max-api");
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

//----------------------------------------
maxApi.post('Hello MaxMSP !\r\n-from Node');
//----------------------------------------

//  Parameters
var FEATURE_NAME 		= [];
var LABEL_NAME 			= [];
var FEATURES 			= {};
var LABELS 				= {};
var DICT 				= {};
var FILTER_FEATURE 		= '';
var FILTER_THRESHOLD	= 0;
var BATCH_SIZE 			= 10;
var EPOCH 				= 100;
var LEARNING_RATE 		= 0.1;
var HIDDEN_LAYERS		= [{units: 2, activation: 'linear'}];
var OPTIMIZER 			= tf.train.sgd;
var LOSS				= 'meanSquaredError';
var METRICS				= 'mse';

// 	Global Variables
var MODEL 				= tf.sequential();
var TENSOR_DATA 		= {};

// 	Constants
const print 			= 	maxApi.post;
const activation_list	=  ['elu', 
							'hardSigmoid', 
							'linear', 
							'relu', 
							'relu6', 
							'selu', 
							'sigmoid', 
							'softmax', 
							'softplus', 
							'softsign', 
							'tanh', 
							'swish', 
							'mish'];
const optimizer_list 	=  [tf.train.sgd,
							tf.train.momentum,
							tf.train.adagrad,
							tf.train.adadelta,
							tf.train.adam,
							tf.train.adamax,
							tf.train.rmsprop];
const loss_list 		=  ['meanSquaredError',
							'meanAbsoluteError',
							'meanAbsolutePercentageError',
							'meanSquaredLogarithmicError',
							'squaredHinge',
							'hinge',
							'categoricalHinge',
							'logcosh',
							'categoricalCrossentropy',
							'sparseCategoricalCrossentropy',
							'binaryCrossentropy',
							'kullbackLeiblerDivergence',
							'poisson',
							'cosineProximity'];
const metrics_list 		=  ['binaryAccuracy',
							'categoricalAccuracy',
							'precision',
							'categoricalCrossentropy',
							'sparseCategoricalCrossentropy',
							'mse',
							'mae',
							'mape',
							'cosine'];
//----------------------------------------

/************************************
*	COMMUNICATION WITH THE MAX API
*************************************/

// 	Max handler for the 'dict' identifier
maxApi.addHandler('dict', async (dict_id) => {
	DICT = await maxApi.getDict(dict_id);
	print("Got Dict");
});

// 	Max handler for the 'feature_name' identifier
maxApi.addHandler('feature_name', (...name) => {
	FEATURE_NAME = [];
	for (let item in name) {FEATURE_NAME.push(name[item])}
});

// 	Max handler for the 'label_name' identifier
maxApi.addHandler('label_name', (...name) => {
	LABEL_NAME = [];
	for (let item in name) {LABEL_NAME.push(name[item])}
});

// 	Max handler for the 'filter_feature' identifier
maxApi.addHandler('filter_feature', async (name) => {
	FILTER_FEATURE = name;
});

// 	Max handler for the 'filter_threshold' identifier
maxApi.addHandler('filter_threshold', async (val) => {
	FILTER_THRESHOLD = val;
});

// 	Max handler for the 'batch_size' identifier
maxApi.addHandler('batch_size', (val) => {
	BATCH_SIZE = parseInt(val);
});

// 	Max handler for the 'epoch' identifier
maxApi.addHandler('epoch', (val) => {
	EPOCH = parseInt(val);
});

// 	Max handler for the 'learning_rate' identifier
maxApi.addHandler('learning_rate', (val) => {
	LEARNING_RATE = val;
});

// 	Max handler for the 'num_hidden' identifier
maxApi.addHandler('num_hidden', (val) => {
	if (parseInt(val) > 0) {
		let layers_to_add = parseInt(val) - HIDDEN_LAYERS.length;
		if (layers_to_add > 0) {
			for (let i = 0; i < layers_to_add; i++) {
				let temp_obj = {units: 2, activation: 'linear'};
				HIDDEN_LAYERS.push(temp_obj);
			}
		} else {
			HIDDEN_LAYERS.length = HIDDEN_LAYERS.length + layers_to_add;
		}
		maxApi.outlet("hidden_layers", HIDDEN_LAYERS);
	} else {
		print("The minimum amount of hidden layers is 1");
	}
});

// 	Max handler for the 'num_units' identifier
maxApi.addHandler('num_units', (val1, val2) => {
	val1 = parseInt(val1);
	val2 = parseInt(val2);
	if (val1 > 0 && val1 <= HIDDEN_LAYERS.length) {
		HIDDEN_LAYERS[val1-1]["units"] = val2;
		maxApi.outlet("hidden_layers", HIDDEN_LAYERS);
	} else {
		print("This hidden layer does not exist");
	}
});

// 	Max handler for the 'activation' identifier
maxApi.addHandler('activation', (val1, val2) => {
	val1 = parseInt(val1);
	val2 = parseInt(val2);
	if (val1 > 0 && val1 <= HIDDEN_LAYERS.length) {
		HIDDEN_LAYERS[val1-1]["activation"] = activation_list[val2];
		maxApi.outlet("hidden_layers", HIDDEN_LAYERS);
	} else {
		print("This hidden layer does not exist");
	}
});

// 	Max handler for the 'optimizer' identifier
maxApi.addHandler('optimizer', (val) => {
	OPTIMIZER = optimizer_list[parseInt(val)];
});

// 	Max handler for the 'loss' identifier
maxApi.addHandler('loss', (val) => {
	LOSS = loss_list[parseInt(val)];
});

// 	Max handler for the 'metrics' identifier
maxApi.addHandler('metrics', (val) => {
	METRICS = metrics_list[parseInt(val)];
});

// 	Max handler for the 'train' identifier
maxApi.addHandler('train', () => {
	train();
});

// 	Max handler for the 'predict' identifier
maxApi.addHandler('predict', (...val) => {    
	predict(val);
});

// 	Max handler for the 'save_model' identifier
maxApi.addHandler('save_model', async (folder_name) => {
	const main_folder = path.resolve(__dirname, "saved_models");
	const model_folder = path.resolve(main_folder, folder_name);

	if (!fs.existsSync(main_folder)) {
		fs.mkdirSync(main_folder, {recursive: true});
	}
	if (!fs.existsSync(model_folder)) {
		fs.mkdirSync(model_folder, {recursive: true});
	}

	await MODEL.save("file://" + model_folder);

	const tensor_data_fMin = TENSOR_DATA.featureMin.arraySync();
	const tensor_data_fMax = TENSOR_DATA.featureMax.arraySync();
	const tensor_data_lMin = TENSOR_DATA.labelMin.arraySync();
	const tensor_data_lMax = TENSOR_DATA.labelMax.arraySync();
	fs.writeFileSync(path.resolve(model_folder, "featureMin.json"), JSON.stringify(tensor_data_fMin), 'utf8');
	fs.writeFileSync(path.resolve(model_folder, "featureMax.json"), JSON.stringify(tensor_data_fMax), 'utf8');
	fs.writeFileSync(path.resolve(model_folder, "labelMin.json"), JSON.stringify(tensor_data_lMin), 'utf8');
	fs.writeFileSync(path.resolve(model_folder, "labelMax.json"), JSON.stringify(tensor_data_lMax), 'utf8');

	print(`Model saved at ${model_folder}`);
});

// 	Max handler for the 'load_model' identifier
maxApi.addHandler('load_model', async (folder_name) => {    
	const main_folder = path.resolve(__dirname, "saved_models");
	const model_folder = path.resolve(main_folder, folder_name);
	const model_file = path.resolve(model_folder, "model.json");

	if (!fs.existsSync(main_folder) || !fs.existsSync(model_folder)) {
		print("Model folder not found.");
	} else {
		MODEL = await tf.loadLayersModel("file://" + model_file);

		const tensor_data_fMin = JSON.parse(fs.readFileSync(path.resolve(model_folder, "featureMin.json"), 'utf8'));
		const tensor_data_fMax = JSON.parse(fs.readFileSync(path.resolve(model_folder, "featureMax.json"), 'utf8'));
		const tensor_data_lMin = JSON.parse(fs.readFileSync(path.resolve(model_folder, "labelMin.json"), 'utf8'));
		const tensor_data_lMax = JSON.parse(fs.readFileSync(path.resolve(model_folder, "labelMax.json"), 'utf8'));
		TENSOR_DATA.featureMin = tf.tensor(tensor_data_fMin);
		TENSOR_DATA.featureMax = tf.tensor(tensor_data_fMax);
		TENSOR_DATA.labelMin = tf.tensor(tensor_data_lMin);
		TENSOR_DATA.labelMax = tf.tensor(tensor_data_lMax);

		print(`Model loaded at ${model_folder}`);		
	}


});
//----------------------------------------

/************************************
*	MULTIPLE REGRESSION ALGORITHM
*************************************/

// 	Formats the given MaxMSP dictionary, creates the model, 
// 	converts the dataset to tensors and trains the model.
async function train() {
	// 	If a dictionary was given
	if (Object.keys(DICT).length > 0) {

		// 	Search and format the given MaxMSP dictionary
		var dataset = clean_standard_dict();

		// 	If a dataset was created
		if (typeof dataset !== 'undefined') {

			print(`Dataset Length:\r\n${Object.keys(dataset).length}`);
			if (FILTER_FEATURE.length > 0) {
				dataset = dataset.filter(d => d.xs[FILTER_FEATURE] >= FILTER_THRESHOLD);

				print(`Filtered Dataset Length:\r\n${Object.keys(dataset).length}`);
				print(`Feature used for filtering:\r\n${FILTER_FEATURE}`);
				print(`Filter threshold:\r\n${FILTER_THRESHOLD}`);
			}

			// 	Convert the dataset to tensors
			TENSOR_DATA = convertToTensor(dataset);
			const {features, labels} = TENSOR_DATA;

			print(`Feature Names:\r\n${FEATURE_NAME}`);
			print(`Label Names:\r\n${LABEL_NAME}`);

			// 	Create the model
			createModel();

			// 	Train the model
			print(`Training with the following parameters:\r\nBatch Size: ${BATCH_SIZE}\r\nEpochs: ${EPOCH}\r\nLearning Rate: ${LEARNING_RATE}\r\nNumber of Hidden Layers: ${HIDDEN_LAYERS.length}`);
			await trainModel(features, labels);
			maxApi.post('Done Training');

			// MODEL.summary(40, 0, x => print(x));
		}
	} else {
		maxApi.post("No dictionary given. Provide a Max MSP dictionary");
	}
}
//----------------------------------------

// 	Search the given MaxMSP dictionary assuming it is in the standard format
function clean_standard_dict() {
	// 	If no feature and label names have been given
	if (FEATURE_NAME.length === 0 && LABEL_NAME.length === 0) {
		let formated = [];
		let check_format = 0;

		// 	Verify if the dictionary is in the standard format
		for (var keys in DICT) {
			if (Object.keys(DICT[keys])[0] !== "features" || Object.keys(DICT[keys])[1] !== "labels") {
				check_format++;
			}
		}

		// 	If the format is standard
		if (check_format === 0) {
			print("Formating the MaxMSP dictionary using the 'clean_standard_dict()' function");
			for (var keys in DICT) {
				// 	Create a list of object from the dictionary in the form :
				// 	[{xs: {feature_name: value}, ys: {label_name: value}}, {...}]
				let temp_obj = {};
				temp_obj["xs"] = DICT[keys]["features"];
				temp_obj["ys"] = DICT[keys]["labels"];
				formated.push(temp_obj);
			}
			// 	Store the names of the features and the labels
			FEATURE_NAME = Object.keys(formated[0]["xs"]);		
			LABEL_NAME = Object.keys(formated[0]["ys"]);
			return formated;
		} else {
			// 	Else, use the function clean_dict()
			print("The format of the given dictionary is not LFO complient, calling 'clean_dict()' function");
			formated = clean_dict(); 
			return formated;
		}
	} else {
		// 	Else, use the function clean_dict()
		print("Feature and/or label names have been given, calling 'clean_dict()' function");
		formated = clean_dict(); 
		return formated;
	}
}

// 	Search the given MaxMSP dictionary for the given feature and label names
// 	and create a list of objects in the form : 
// 	[{xs: {feature_name: value}, ys: {label_name: value}}, {...}]
function clean_dict() {
	// 	If the feature and label names have been given
	if (FEATURE_NAME.length > 0 && LABEL_NAME.length > 0) {
		var names = FEATURE_NAME.concat(LABEL_NAME);
		var formated = [];
		var iter = 0;

		for (var item in names) {
			iter = 0;
			// 	If the name is a feature, use the identifier 'xs'
			if (FEATURE_NAME.includes(names[item])) {
				var identifier = "xs";
				search_dict(DICT, names[item], identifier);
			// 	Else use the identifier 'ys'
			} else {
				var identifier = "ys";
				search_dict(DICT, names[item], identifier);
			}
		}

		function search_dict(obj, name, identifier) {
			// 	Iterate the object using for..in
			for (var keys in obj) {
				// 	Check if the object has any property by that name 
				if (obj.hasOwnProperty(keys) && typeof obj[keys] === 'object') {
					// 	If the key is not undefined, get it's value
					if (obj[keys][name] !== undefined) {
						// 	If the index in the list 'formated' is not created
						if (typeof formated[iter] === 'undefined') {
							// Create and push the objects "xs" and "ys"
							var struct_temp = {};						
							struct_temp["xs"] = {};
							struct_temp["ys"] = {};
							formated.push(struct_temp);
							// Create and push the found key:value to the appropriate identifier
							var obj_temp = {};
							obj_temp[name] = obj[keys][name];
							formated[iter][identifier] = obj_temp;
						// 	If the index is already created, add the key:value at the index
						} else {
							formated[iter][identifier][name] = obj[keys][name];
						}
						iter++;
					} else {
						// 	Else call the function again using the new object value
						search_dict(obj[keys], name, identifier);
					}
				}
			}
		}
		return formated;
	} else {
		print("Missing feature and/or label names");
	}
}
//----------------------------------------

function convertToTensor(data) {
	//	Wrapping these calculations in a tidy will dispose any intermediate tensors.
	return tf.tidy(() => {
		//	Shuffle the data
		tf.util.shuffle(data);

		//	Convert the data to Tensor :
		// 	Create an array for the features examples 
		const features = data.map(d => Object.values(d.xs));
		// 	Create an array for the labels examples
		const labels = data.map(d => Object.values(d.ys));

		// 	Convert each array data to a 2d tensor (values, shape: [x, y])
		const featureTensor = tf.tensor2d(features, [features.length, FEATURE_NAME.length]);
		const labelTensor = tf.tensor2d(labels, [labels.length, LABEL_NAME.length]);

		//	Normalize the data to the range 0 - 1 using min-max scaling
		const featureMax = featureTensor.max();
		const featureMin = featureTensor.min();
		const labelMax = labelTensor.max();
		const labelMin = labelTensor.min();

		const normalizedFeatures = featureTensor.sub(featureMin).div(featureMax.sub(featureMin));
		const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

		return {
			features: 	normalizedFeatures,
			labels: 	normalizedLabels,
			// Return the min/max bounds for the prediction step
			featureMax,
			featureMin,
			labelMax,
			labelMin,
		}
	});
}
//----------------------------------------

//	Create the model according to the information in the 'HIDDEN_LAYERS' dictionary
function createModel() {
	MODEL = tf.sequential();

	// Input layer and first hidden layer
	MODEL.add(tf.layers.dense({				inputShape: [FEATURE_NAME.length], 
											units: 		HIDDEN_LAYERS[0]["units"], 
											activation: HIDDEN_LAYERS[0]["activation"]}));

	if (HIDDEN_LAYERS.length > 1) {
		for (let item in HIDDEN_LAYERS) {
			if (item > 0) {
				// Additional hidden layers
				MODEL.add(tf.layers.dense({	units: 		HIDDEN_LAYERS[item]["units"], 
											activation: HIDDEN_LAYERS[item]["activation"]}));
			}
		}
	}
	// Output layer
	MODEL.add(tf.layers.dense({				units: 		LABEL_NAME.length, 
											activation: 'linear'}));
}
//----------------------------------------

async function trainModel(features, labels) {
	// 	Compile the model for training
	MODEL.compile({
		optimizer: OPTIMIZER(LEARNING_RATE),
		loss: LOSS,
		metrics:[METRICS]
	});

	// 	Fit the dataset
	return await MODEL.fit(features, labels, {
		batchSize: BATCH_SIZE,
		epochs: EPOCH,
		shuffle: true,
		callbacks: {onEpochEnd: async (epoch, logs) => 
						{
							maxApi.outlet("epoch", epoch);
							maxApi.outlet("loss", logs["loss"]);
							maxApi.outlet("metrics", logs[METRICS]);
                		}
					}
	});
}
//----------------------------------------

//  If the number of values received is the same size as the input layer of the model:
//  Output a prediction to outlet_#1 in the format [predicted value_1 value_2 ... value_x] 
function predict(input) {
	// If MODEL has layers
	if (MODEL.layers.length > 0) {

		let model_input_size = MODEL.layers[0].getConfig().batchInputShape[1];

		if (input.length === model_input_size) {
			const {featureMin, featureMax, labelMin, labelMax} = TENSOR_DATA;

			const preds = tf.tidy(() => {
				//	Create a tensor from the given exemple
				const inputTensor = tf.tensor2d(input, [1, model_input_size]);
				// 	Create a normalized tensor
				const normalizedInput = inputTensor.sub(featureMin).div(featureMax.sub(featureMin));
				//	Feed the example to the model
				const preds = MODEL.predict(normalizedInput);
				//	Unormalize the prediction
				const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

				return unNormPreds;
			});

			//	Output the prediction to the outlet with the identifier 'predicted'
			preds.array().then(array => maxApi.outlet("predicted", ...array[0]));

	    } else {
	        maxApi.post("The input list length is not the right size");
	    }
    } else {
    	maxApi.post("There's no model yet ..");
	}
}