// // console.log("hello tensorflow");

// import {MnistData} from './data.js';

// async function showExamples(data) {
//     //创造visor的容器
//     const surface =tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});

//     //得到例子
//     const examples = data.nextTestBatch(20);
//     const numExamples = examples.xs.shape[0];
  
//     //创建canvas来渲染每个例子
//     for (let i = 0; i < numExamples; i++) {
//         const imageTensor = tf.tidy(() => {
//           // Reshape the image to 28x28 px
//           return examples.xs
//             .slice([i, 0], [1, examples.xs.shape[1]])
//             .reshape([28, 28, 1]);
//         });

//         const canvas = document.createElement('canvas');
//         canvas.width = 28;
//         canvas.height = 28;
//         canvas.style = 'margin: 4px;';
//         await tf.browser.toPixels(imageTensor, canvas);
//         surface.drawArea.appendChild(canvas);
    
//         imageTensor.dispose();
// }
// }

// async function run() {
//     const data = new MnistData();
//     await data.load();
//     await showExamples(data);
//   }

// document.addEventListener("DOMContentLoaded",run);
import {MnistData} from './data.js';

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);


    const model = getModel();
    tfvis.show.modelSummary({name:'model arch',tab:"model"},model);
    await train(model,data);
    //显示评估
    await showAccuracy(model, data);
    await showConfusion(model, data);
}

document.addEventListener('DOMContentLoaded', run);

//定义模型架构
function getModel(){
    const model = tf.sequential();
    const IMG_WIDTH = 28;
    const IMG_HEIGHT = 28;
    const IMG_CHANNELS = 1;

    // 在我們的捲積神經網絡的第一層，我們有
    // 指定輸入形狀。然後我們指定一些參數
    // 在這一層發生的捲積操作
    model.add(tf.layers.conv2d({
        inputShape:[IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS],
        kernelSize:5,
        filters:8,
        strides:1,
        activation:'relu',
        kernelInitializer:"varianceScaling"

    }));

    //MaxPooling 層作為一種使用最大值的下採樣而不是平均
    model.add(tf.layers.conv2d({
        kernelSize:5,
        filters:16,
        strides:1,
        activation:'relu',
        kernelInitializer:'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize:[2,2],strides:[2,2]}));
    // 現在我們將 2D 過濾器的輸出展平為 1D 向量以準備
    // 它用於輸入到我們的最後一層。這是餵食時的常見做法
    // 更高維數據到最終分類輸出層。
    model.add(tf.layers.flatten());

    //最后一层是dense层，有10个输出
    const NUM_OUTPUT_CLASSES = 10;
    //计算最终的概率分布
    model.add(tf.layers.dense({
        units:NUM_OUTPUT_CLASSES,
        kernelInitializer:'varianceScaling',
        activation:'softmax'
    }));

    //选择编译器，损失函数等，编译模型然后返回
    const optimizer = tf.train.adam();

    model.compile({
        optimizer:optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;



}

//训练模型
async function train(model,data){
    //监控指标
    const metrics = ['loss','val_loss','acc','val_acc'];
    const container = {
        name:'model training',
        tab:"model",
        styles:{height:'1000px'}
    };
    
    const fitCallbacks = tfvis.show.fitCallbacks(container,metrics);

    const BATCH_SIZE = 512;
    //将 trainDataSize 设置为 5500 并将 testDataSize 设置为 1000 可加快实验速度
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE=1000;

    const [trainXs,trainYs] = tf.tidy(()=>{
        const d = data.nextTestBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE,28,28,1]),
            d.labels
        ];
    });

    const [testXs,testYs] = tf.tidy(()=>{
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE,28,28,1]),
            d.labels
        ];
    });
    return model.fit(trainXs,trainYs,{
        batchSize:BATCH_SIZE,
        ValidityState:[testXs,testYs],
        epochs:10,
        shuffle:true,
        callbacks:fitCallbacks
    });
}

//评估模型
const classNames = ['zero','one','two','three','four','five','six','seven','eight','nine'];

//做出预测
function doPrediction(model,data,testDataSize=500){
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize,IMAGE_WIDTH,IMAGE_HEIGHT,1]);
    //argmax 函数为我们提供概率最高的类的索引。请注意，模型会输出每个类的概率。我们会找出最高概率，并指定将其用作预测。
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds,labels];
}

//显示每个类的准确率
async function showAccuracy(model,data){
    const [preds,labels] = doPrediction(model,data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels,preds);
    const container = {name:"Accuracy",tab:"Evaluation"};
    tfvis.show.perClassAccuracy(container,classAccuracy,classNames);

    labels.dispose();
}
//显示混淆矩阵
async function showConfusion(model,data){
    const [preds,labels] = doPrediction(model,data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels,preds);
    const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
    tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});
  
    labels.dispose();
}