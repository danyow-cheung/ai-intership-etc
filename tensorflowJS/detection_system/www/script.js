import * as DICTIONARY from "./dictionary.js";
// import * as io from "../server.js";
// ML 模型期望的輸入元素的數量。
const ENCODING_LENGTH = 20;


/**
 * 接受單詞數組，將單詞轉換為標記的函數，
 * 然後返回標記化的張量表示
 * 可用作機器學習模型的輸入。
 */
 function tokenize(wordArray){
  //以【start】token开始
  let returnArray = [DICTIONARY.START];
  
  // 循環遍歷要編碼的句子中的單詞。
  // 如果在字典中找到單詞，則添加該數字
  // 您添加 UNKNOWN 令牌。
  for (var i = 0; i < wordArray.length; i++) {
      let encoding = DICTIONARY.LOOKUP[wordArray[i]];
      returnArray.push(encoding === undefined ? DICTIONARY.UNKNOWN : encoding);
  }
  // 最後如果字數 < 最小編碼長度
  // 減 1（由於開始標記），用 PAD 標記填充其餘部分。
  while (returnArray.length < ENCODING_LENGTH ) {
      returnArray.push(DICTIONARY.PAD);
  }
  //打印日志去看发生了什么
  console.log([returnArray]);

  //转换成tenors然后返回
  return tf.tensor([returnArray]);
}



/*引用主要dom元素*/
const POST_COMMENT_BTN = document.getElementById('post');
const COMMENT_TEXT = document.getElementById('comment');
const COMMENTS_LIST = document.getElementById('commentsList');

// CSS 樣式類來指示註釋正在被處理時
// 發布以向用戶提供視覺反饋。
const PROCESSING_CLASS = 'processing';
// 存儲登錄用戶的用戶名。現在你沒有身份驗證
// 所以默認為匿名直到知道。
var currentUserName = 'Anonymous';


/*处理评论发布*/
//处理发布评论的函数
function handleCommentPost(){
    //只有当你没发评论之前才会继续
    if (! POST_COMMENT_BTN.classList.contains(PROCESSING_CLASS)) {
        // Set styles to show processing in case it takes a long time.
        POST_COMMENT_BTN.classList.add(PROCESSING_CLASS);
        COMMENT_TEXT.classList.add(PROCESSING_CLASS);
        //从dom获取到输入的信息
        let currentComment = COMMENT_TEXT.innerText;

        // 將句子轉換為 ML 模型期望的小寫
        // 去除所有非字母數字或空格的字符
        // 然後在空格上拆分以創建一個單詞數組
        let lowercaseSentenceArray = currentComment.toLowerCase().replace(/[^\w\s]/g, ' ').split(' ');
        

        // 在內存中創建一個列表項 DOM 元素。
        let li = document.createElement("li");

        // 記住 loadAndPredict 是異步的，所以你使用 then
        // 在繼續之前等待結果的關鍵字。
        loadAndPredict(tokenize(lowercaseSentenceArray), li).then(function() {
            //重置类别，准备下一个输入
            POST_COMMENT_BTN.classList.remove(PROCESSING_CLASS);
            COMMENT_TEXT.classList.remove(PROCESSING_CLASS);
      
            let p = document.createElement('p');
            p.innerText = COMMENT_TEXT.innerText;
      
            let spanName = document.createElement('span');
            spanName.setAttribute('class', 'username');
            spanName.innerText = currentUserName;
      
            let spanDate = document.createElement('span');
            spanDate.setAttribute('class', 'timestamp');
            let curDate = new Date();
            spanDate.innerText = curDate.toLocaleString();
      
            li.appendChild(spanName);
            li.appendChild(spanDate);
            li.appendChild(p);
            COMMENTS_LIST.prepend(li);
      
             //重设评论内容
            COMMENT_TEXT.innerText = '';
          });
        console.log(currentComment);

    }
}

POST_COMMENT_BTN.addEventListener("click",handleCommentPost);


//设置model.json的路径
const MODEL_JSON_URL = 'model.json';
//設置垃圾評論被標記的最小置信度。
// 記住這是一個從 0 到 1 的數字，代表一個百分比
// 所以這裡 0.75 == 75% 確定它是垃圾郵件。
const SPAM_THRESHOLD = 0.75;


// 創建一個變量來存儲加載的模型，一旦它準備好了
// 你可以稍後在程序的其他地方使用它。
var model = undefined;


/**
 * 異步函數加載TFJS模型，然後使用它
 * 預測輸入是否為垃圾郵件。
 */
async function loadAndPredict(inputTensor,domComment){
    //记载model.json和二进制文件，请注意这是异步操作所以可以用awaut的关键字
    if (model === undefined) {
        model = await tf.loadLayersModel(MODEL_JSON_URL);
    }

    //一旦模型加载之后，就可以用模型来预测
    //传入数据之后，还可以存储结果
    var results = await model.predict(inputTensor);

    //打印结果
    results.print();

    results.data().then((dataArray)=>{
        if (dataArray[1] > SPAM_THRESHOLD) {
          domComment.classList.add('spam');
        } else {
          // Emit socket.io comment event for server to handle containing
          // all the comment data you would need to render the comment on
          // a remote client's front end.
          socket.emit('comment', {
            username: currentUserName,
            timestamp: domComment.querySelectorAll('span')[1].innerText,
            comment: domComment.querySelectorAll('p')[0].innerText
          });
        }
      })
    }
//因此，在本例中，模型表示99.96011% （在结果对象中显示为 0.9996011）您已提供的输入（其值[1,3,12,18,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]是非垃圾邮件（即 False）。
// loadAndPredict(tf.tensor([[1,3,12,18,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]));





// Connect to Socket.io on the Node.js backend.
var socket = io.connect();

function handleRemoteComments(data) {
  // Render a new comment to DOM from a remote client.
  let li = document.createElement('li');
  let p = document.createElement('p');
  p.innerText = data.comment;

  let spanName = document.createElement('span');
  spanName.setAttribute('class', 'username');
  spanName.innerText = data.username;

  let spanDate = document.createElement('span');
  spanDate.setAttribute('class', 'timestamp');
  spanDate.innerText = data.timestamp;

  li.appendChild(spanName);
  li.appendChild(spanDate);
  li.appendChild(p);

  COMMENTS_LIST.prepend(li);
}

// Add event listener to receive remote comments that passed
// spam check.
socket.on('remoteComment', handleRemoteComments);

