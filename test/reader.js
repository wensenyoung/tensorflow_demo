const fs = require("fs");
const tf = require("@tensorflow/tfjs");

function readImageFile(buffer) {
    const magic = buffer.readInt32BE(0);
    const size = buffer.readInt32BE(4);
    const rows = buffer.readInt32BE(8);
    const cols = buffer.readInt32BE(12);

    let offset = 16;
    const image_size = rows * cols;


    return tf.tidy(() => {
        return tf.tensor2d(buffer.slice(offset), [size, image_size]);
    })
}


function readLabelFile(buffer, oneHot = false) {
    const magic = buffer.readInt32BE(0);
    const size = buffer.readInt32BE(4);

    let offset = 8;
    let ret = Array.from(buffer.slice(offset));

    return tf.tidy(() => {
        if (oneHot) {
            return tf.tensor1d(ret, "int32").oneHot(10).toFloat();
        } else {
            return tf.tensor1d(ret);
        }
    })

}

function getTestsData(oneHot = true) {
    let imageBuffer = fs.readFileSync("../data/t10k-images.idx3-ubyte");
    let labelBuffer = fs.readFileSync("../data/t10k-labels.idx1-ubyte");

    let images = readImageFile(imageBuffer);
    let labels = readLabelFile(labelBuffer, oneHot);

    return {
        images: images,
        labels: labels
    }
}

function getTransData(oneHot = true) {
    let imageBuffer = fs.readFileSync("../data/t10k-images.idx3-ubyte");
    let labelBuffer = fs.readFileSync("../data/t10k-labels.idx1-ubyte");

    let images = readImageFile(imageBuffer);
    let labels = readLabelFile(labelBuffer, oneHot);

    return {
        images: images,
        labels: labels
    }
}


async function checkPersistent() {
    let path = `file://${__dirname}/../data/huobi_BTC_History`;
    let data = tf.data.csv(path);

    data.forEachAsync(item => {
        if (!previous) {
            previous = item;
        } else {
            if (previous.time + 60 !== item.time) {
                console.warn('find error data id = ', previous.time)
            }
            previous = item;
        }
    })
}

let counter = {};

function formatData() {
    let path = `file://${__dirname}/../data/huobi_BTC_History`;
    let data = tf.data.csv(path);

    const history_length = 60;
    const predict_length = 5;

    const limit_length = history_length + predict_length;

    let temp = [];
    let ret = {xs: [], ys: []};

    return new Promise(resolve => {
        data.forEachAsync(item => {
            temp.push(item);

            if (temp.length > limit_length) {
                // 移除最老的数据
                temp.shift();

                // 单次训练数据集
                // const [open, high, low, close, vol, count] = [0, 1, 2, 3, 4, 5];
                let train_data = [];

                for (let i = 0; i < history_length; i++) {
                    let t = temp[i];
                    train_data.push(t.open);
                    train_data.push(t.high);
                    train_data.push(t.low);
                    train_data.push(t.close);
                    train_data.push(t.amount);
                    train_data.push(t.count);
                }

                // 结果集
                // todo 如果一个训练结果同时满足多个结果怎么办？
                // todo 比如1分钟涨幅> 1% 2分钟涨幅也>1%, 就会出现一个输入对应多个正确的结果
                let label_data = formatLabel(temp.slice(history_length));
                ret.xs.push(Buffer.from(train_data));
                ret.ys.push(Buffer.from(label_data));
            }
        }).then(()=>{
            resolve(ret);
        });
    });



    // 按照最高价除以当前价格
    function formatLabel(datas) {
        let mergeKline = mergeData(datas);

        let label = [];

        let assert = (val, left, right) => (val >= left && val < right) ? 1 : 0;

        let high_percent = (mergeKline.high - mergeKline.open) / mergeKline.open;
        // let close_percent = (mergeKline.close - mergeKline.open) / mergeKline.open;
        // let low_percent = (mergeKline.low - mergeKline.open) / mergeKline.open;


        // label.push(assert(low_percent, -Infinity, -0.020));
        // label.push(assert(low_percent, -0.020, -0.018));
        // label.push(assert(low_percent, -0.018, -0.016));
        // label.push(assert(low_percent, -0.016, -0.014));
        // label.push(assert(low_percent, -0.014, -0.012));
        // label.push(assert(low_percent, -0.012, -0.010));
        // label.push(assert(low_percent, -0.010, -0.008));
        // label.push(assert(low_percent, -0.008, -0.006));
        // label.push(assert(low_percent, -0.006, -0.004));
        // label.push(assert(low_percent, -0.004, -0.002));
        // label.push(assert(low_percent, -0.002, 0));

        label.push(assert(high_percent, 0, 0.002));
        label.push(assert(high_percent, 0.002, 0.004));
        label.push(assert(high_percent, 0.004, 0.006));
        label.push(assert(high_percent, 0.006, 0.008));
        label.push(assert(high_percent, 0.008, 0.010));
        label.push(assert(high_percent, 0.010, 0.012));
        label.push(assert(high_percent, 0.012, 0.014));
        label.push(assert(high_percent, 0.014, 0.018));
        label.push(assert(high_percent, 0.018, Infinity));

        let index = label.indexOf(1);
        counter[index] = (counter[index] || 0) + 1;
        console.log(JSON.stringify(counter));

        return label;
    }

    // 合并kline
    function mergeData(datas) {
        let data = {};
        let len = datas.length;
        data.open = datas[0].open;
        data.close = datas[len - 1].close;

        for (let item of datas) {
            if (item.high > (data.high || 0)) {
                data.high = item.high;
            }

            if (item.low < (data.low || Infinity)) {
                data.low = item.low;
            }
        }

        return data;
    }
}

(async ()=>{
    let ret = await formatData();
    let tensor_xs = tf.tensor2d(ret.xs);
    let tensor_ys = tf.tensor2d(ret.ys);
    // console.log(ret.xs.length, ret.ys.length);
})();

module.exports = {
    getTestsData,
    getTransData
};
