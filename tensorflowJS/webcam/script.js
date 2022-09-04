const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const demoSection = document.getElementById("demos");
const enableWebcamButton = document.getElementById("webcamButton");


//检查webcam是否可用
function getUserMediaSupported(){
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// 如果支持網絡攝像頭，則為用戶添加事件監聽器到按鈕
// 想要激活它來調用 enableCam 函數，我們將
// 在下一步中定義。
if (getUserMediaSupported()){
    enableWebcamButton.addEventListener("click",enableCam);
}else{
    console.warn('getUserMedia() is not supported by your browser');
}

// 啟用實時網絡攝像頭視圖並開始分類。
function enableCam(event){
    // 只有在 COCO-SSD 完成加載後才繼續。
    if (!model){
        return;
    }

    //点击按钮之后隐藏按钮
    event.target.classList.add("removed");

    //getUsermedia 参数强迫去开视频
    const constraints = {
        video:true
    };
    
    //激活webcam数据流
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream){
        video.srcObject = stream;
        video.addEventListener("loadeddata",predictWebcam);
    });

}



var model = undefined;
// 在我們可以使用 COCO-SSD 類之前，我們必須等待它完成
// 加載。機器學習模型可能很大並且需要一些時間
// 獲取運行所需的一切。
// 注意：cocoSsd 是從我們的 index.html 加載的外部對象
// 腳本標籤導入，因此忽略 Glitch 中的任何警告。
cocoSsd.load().then(function(loadedModel){
    model =loadedModel;
    //展示deno现在模型可用了
    demoSection.classList.remove("invisible");
});

var children = [];
function predictWebcam(){
    //预测每一帧中的内容
    model.detect(video).then(function(predictions){
        //移除前几帧的内容
        for(let i = 0;i<children.length;i++){
            liveView.removeChild(children[i]);
        }
        children.splice(0);
        // 現在讓我們循環遍歷預測並將它們繪製到實時視圖，如果
        // 他們的置信度得分很高。
        for(let n =0;n<predictions.length;n++){
            //超过66%认定是对的并且显示
            if(predictions[n].score>0.66){
                const p = document.createElement("p");
                p.innerText = predictions[n].class  + ' - with ' 
                                                    + Math.round(parseFloat(predictions[n].score) * 100) 
                                                    + '% confidence.';
                p.style = 'margin-left: ' + predictions[n].bbox[0] + 'px; margin-top: '
                                            + (predictions[n].bbox[1] - 10) + 'px; width: ' 
                                            + (predictions[n].bbox[2] - 10) + 'px; top: 0; left: 0;';
                
                const highlighter = document.createElement('div');
                highlighter.setAttribute('class', 'highlighter');

                
                highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: '
                                                + predictions[n].bbox[1] + 'px; width: ' 
                                                + predictions[n].bbox[2] + 'px; height: '
                                                + predictions[n].bbox[3] + 'px;';
                                    
                liveView.appendChild(highlighter);
                liveView.appendChild(p);
                                            
                // Store drawn objects in memory so we can delete them next time around.
                children.push(highlighter);
                children.push(p);
            }
        }
    // 再次調用此函數以繼續預測瀏覽器何時準備就緒。        
    window.requestAnimationFrame(predictWebcam);
    });
}