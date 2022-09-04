//加载数据，设置格式并直观呈现



/**
 * 將汽車數據簡化為我們感興趣的變量
 * 並清除丟失的數據。
 */
async function getData(){
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));
  
    return cleaned;
}

async function run(){
    //加载和打印原始输入数据
    const data = await getData();
    const values = data.map(d=>({
        x:d.horsepower,
        y:d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
          xLabel: 'Horsepower',
          yLabel: 'MPG',
          height: 300
        }
    );

    //剩余代码

    const model = createModel();
    tfvis.show.modelSummary({name:"model summary"},model);


    //转化数据用于训练
    const tensorData = convertToTensor(data);
    const {inputs,labels}=tensorData;

    //训练模型
    await trainModel(model,inputs,labels);
    console.log("done training");


    // 使用模型進行一些預測並將它們與
    //原始數據
    testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);


//定义模型架构
function createModel(){
    const model = tf.sequential();

    //添加单输入层
    model.add(tf.layers.dense({inputShape:[1],units:1,useBias:true}));

    //添加输出层
    model.add(tf.layers.dense({units:1,useBias:true}));

    return model;
}

//准备数据以用于训练
/**
 * 將輸入數據轉換為我們可以用於機器的張量
 * 學習。我們還將做_shuffling_的重要最佳實踐
 * 數據和_規範化_數據
 * y 軸上的 MPG。
 */
function convertToTensor(data){
    // 將這些計算整齊地包裝起來將處理任何
    // 中間張量
    return tf.tidy(()=>{
        //打乱数据
        tf.util.shuffle(data);

        //转换数据到张量
        const inputs = data.map(d=>d.horsepower);
        const labels = data.map(d=>d.mpg);

        const inputTensor = tf.tensor2d(inputs,[inputs.length,1]);
        const labelTensor = tf.tensor2d(labels,[labels.length,1]);

        //归一化数据使用min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizeInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizeLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs:normalizeInputs,
            labels:normalizeLabels,
            // 返回最小最大邊界，以便我們稍後使用它們。
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });

}

//训练模型

async function trainModel(model,inputs,labels){
    //准备训练的模型
    model.compile({
        optimizer:tf.train.adam(),
        loss:tf.losses.meanSquaredError,
        metrics:['mse'],
    });

    const batchSize = 32;
    const epochs  = 50;
    //启动训练循环
    return await model.fit(inputs,labels,{
        batchSize,
        epochs,
        shuffle:true,
        callbacks:tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}
    

//做出预测
function testModel(model,inputData,normalizationData){
    const {inputMax,inputMin,labelMin,labelMax}=normalizationData;
    // 為 0 到 1 之間的統一數字範圍生成預測；
    // 我們通過與 min-max 縮放相反的方式對數據進行非標準化
    // 我們之前做的。
    const [xs,preds] = tf.tidy(()=>{
        const xs = tf.linspace(0,1,100);
        const preds = model.predict(xs.reshape([100,1]));

        const unNormXs = xs.mul(inputMax.sub(inputMin))
                           .add(inputMin);
                           
        const unNormPreds = preds.mul(labelMax.sub(labelMin))
                                .add(labelMin);
        
        //没有归一化的数据分析
        //要将数据恢复到原始范围（而非 0-1），我们会使用归一化过程中计算的值，
        //但只是进行逆运算。
        return [unNormXs.dataSync(),unNormPreds.dataSync()];

    });

    const predictedPoints = Array.from(xs).map((val,i)=>{
        return {x:val,y:preds[i]}
    });

    const originalPoints = inputData.map(d=>({
        x:d.horsepower,
        y:d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
    );
}