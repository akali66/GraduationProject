import sqlite3

def write():
    with open('DEFENSE_PROJECT_MASTERY_CN.md', 'r', encoding='utf-8') as f:
        text = f.read()

    append_text = '''
---

## 8. 系统工程与后端框架核心考点 (FastAPI / 架构 / 前端)
在系统开发中，除了计算机视觉算法，工程化落地同样是重头戏（且容易被问到）。以下是系统框架层面的核心知识点。

### 8.1 为什么选择 FastAPI？异步与依赖注入
**代码位置**：`app.py`
```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Model Web API")

@app.post("/api/detect")
async def detect_image(
    file: UploadFile = File(...),
    method: str = Form("method1"),
    ...
):
```
* **FastAPI 框架优势**：高性能（基于 Starlette 和 Pydantic）、自带类型提示和数据校验、自动生成接口文档（Swagger UI）。相比 Flask 或 Django，它在构建纯算法 API 时更轻量且极其快速。
* **异步 (async/await)**：`async def detect_image` 表示这是一个异步函数。在处理图像上传这种可能产生 I/O 阻塞的操作（如网络传输、磁盘读取）时，服务器可以把等待的时间拿去处理其他用户的请求，极大提高了并发响应能力。
* **表单参数与依赖注入 (UploadFile/Form)**：`file: UploadFile = File(...)` 表示框架会自动从表单中拦截名为 `file` 的二进制文件流，并注入到函数变量中，省去了手动解析 HTTP 请求头的繁琐工作。

### 8.2 跨域资源共享 (CORS) 是什么？
**代码位置**：`app.py`
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
> **💡术语解释——CORS (Cross-Origin Resource Sharing)**：浏览器的同源策略会阻止网页向不同端口或域名的服务器发请求。前端在 `index.html` (可能直接双击打开，协议是 `file://` 或 `http://localhost:5500`)，而后端运行在 `127.0.0.1:8000`。加入 CORS 中间件并设为 `*` (允许所有)，就是为了让前端能顺畅跨网段呼叫后端算法。

### 8.3 图像传输方案：为何用 Base64？
**代码位置**：`app.py` (`img_to_base64`函数)
```python
def img_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')
```
* 当算法将圆画在图片上之后，系统并没有把结果 `cv2.imwrite` 保存到服务器硬盘生成一个 URL 给前端，而是直接将内存里的图片压缩成了 PNG 格式的纯文本编码（Base64 字符串），然后直接塞进 JSON 返回。
* **优势**：**无状态与零 I/O 磁盘消耗**。图片不再占用服务器硬盘，不会产生垃圾文件，前端拿到长串代码 `href="data:image/png;base64,xxx..."` 就能直接在 `<img>` 标签里渲染展示。

### 8.4 设计模式：单例模式加载大型模型
**代码位置**：`detectors.py`
```python
_yolo_model = None

def get_yolo_model(model_path='...'):
    global _yolo_model
    if _yolo_model is None:
        if YOLO is not None:
            _yolo_model = YOLO(model_path)  # 仅在第一次真正加载
    return _yolo_model
```
> **💡术语解释——单例模式 (Singleton Pattern)**：YOLO 模型文件几十兆，加载到内存和显存需要 1-3 秒。如果每一次用户点击“检测”都去 `YOLO(model_path)`，不仅卡顿，还会把显存撑爆（OOM）。通过全局变量 `_yolo_model` 作判断，保证了整个项目的生命周期内，模型只会在第一次被使用时加载一次。

### 8.5 前端工程核心总结
**代码位置**：`index.html`
* **Tailwind CSS**：前端标签里满是 `class="flex items-center justify-between px-4..."`。这是使用了 Tailwind 原子化 CSS 框架。它摒弃了传统的单独写一个 `.css` 文件，而是将样式拆解成了无数个原子类，极大地提高了 UI 开发速度，让界面现代化、自适应且代码紧凑。
* **Fetch API**：替代老旧的 `$.ajax` 或 `XMLHttpRequest`，通过 `async/await` 配合，极其优雅地完成了向后端的异步数据提交，并且自动处理 JSON 反序列化。
'''

    with open('DEFENSE_PROJECT_MASTERY_CN.md', 'w', encoding='utf-8') as f:
        f.write(text + append_text)

write()
