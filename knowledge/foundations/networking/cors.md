---
title: "CORS：跨源资源共享"
type: concept
domain: foundations
status: evergreen
---

# CORS：跨源资源共享

## 一句话解释

CORS（Cross-Origin Resource Sharing，跨源资源共享）是一套由服务器通过 HTTP 响应头声明许可、由浏览器执行的机制，让网页脚本可以读取获准的跨源响应。

## 详细解释

普通 HTTP(S) URL 的“源”由协议、主机和有效端口决定，任一不同就是跨源，路径不同不影响同源。例如，`http://localhost:3000` 请求 `http://localhost:8080/api/users` 就是跨源。中文常说的“跨域”在这里准确含义是“跨源”。

浏览器的同源策略限制脚本读取跨源资源。CORS 为这种读取提供受控许可；服务器返回 HTTP 200，并不意味着浏览器一定允许 JavaScript 读取响应。

## 工作原理

1. 浏览器发送跨源 CORS 请求时，通过 `Origin` 表明请求来源。
2. 满足安全列出的方法、请求头及其值等条件的请求，可直接发送，常称为“简单请求”。使用 `PUT`、`Content-Type: application/json` 或 `Authorization` 等通常需要先发送 `OPTIONS` 预检；有效的预检缓存可省去重复预检。
3. 预检通过 `Access-Control-Request-Method` 和必要的 `Access-Control-Request-Headers` 询问许可，服务器通过 `Access-Control-Allow-Origin`、`Access-Control-Allow-Methods` 和 `Access-Control-Allow-Headers` 回答。
4. 预检获准后才发送实际请求；实际响应仍须通过 CORS 检查，才能交给脚本读取。无需预检的请求也要检查实际响应。

例如，允许 `http://localhost:3000` 读取接口响应，服务器在实际响应中返回：

```http
Access-Control-Allow-Origin: http://localhost:3000
```

若请求使用 `credentials: "include"`，服务器还需返回 `Access-Control-Allow-Credentials: true`，且允许来源必须是具体源，不能是 `*`。Cookie 是否实际发送仍受 SameSite 和浏览器 Cookie 策略等约束。[协议与凭据规则](https://fetch.spec.whatwg.org/#http-cors-protocol)

## 适用边界

- CORS 由浏览器执行，`curl` 或服务器间调用通常不受它限制。
- CORS 不保证请求未到达服务器：无需预检的请求可能已执行，只是响应不可读。因此它不能替代身份认证、权限校验或 CSRF 防护。
- `fetch` 的 `mode: "no-cors"` 不会赋予跨源读取权限；返回的不透明响应无法由脚本读取正文和状态。

## 实践意义

- 在后端或网关配置允许来源；前端自行添加 `Access-Control-Allow-Origin` 请求头无效。
- 排查时分别检查 OPTIONS 和实际响应，错误响应也应带上适用的 CORS 头。
- 若依据请求来源动态返回许可，应先校验来源白名单，并设置 `Vary: Origin`，避免缓存混用不同来源的响应。

## 参考资料

- [WHATWG Fetch：CORS protocol](https://fetch.spec.whatwg.org/#http-cors-protocol)
- [WHATWG Fetch：CORS protocol and credentials](https://fetch.spec.whatwg.org/#cors-protocol-and-credentials)
- [WHATWG Fetch：CORS protocol and HTTP caches](https://fetch.spec.whatwg.org/#cors-protocol-and-http-caches)
