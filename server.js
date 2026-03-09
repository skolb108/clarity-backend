import http from "http";

const PORT = process.env.PORT || 8080;

const server = http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("Railway server working");
});

server.listen(PORT, "0.0.0.0", () => {
  console.log("Server running on port", PORT);
});