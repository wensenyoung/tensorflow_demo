const tf = require("@tensorflow/tfjs");
const {getTestsData, getTransData} = require("./reader");


function createModel(){
    const model = tf.sequential();
    const output_classes = 10;

    model.add(tf.layers.dense({
        batchInputShape: [100, 784],
        units: output_classes,
        kernelInitializer: "varianceScaling",
        activation: "softmax"
    }));

    const optimizer = tf.train.adam(0.001);

    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    return model;
}
function train(model) {
    const {images, labels} = getTransData(true);
    const test_data = getTestsData(true);

    model.fit(images, labels, {
        epochs: 100,
        validationData: [test_data.images, test_data.labels],
        batchSize: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(epoch, format_output(logs));
            }
        }
    });
}

function format_output(obj){
    return `loss: ${obj['loss']}, acc: ${obj['acc']}, val_loss: ${obj['val_loss']}, val_acc: ${obj['val_acc']}`;
}

function run(){
    let model = createModel();
    train(model)
}


run();
