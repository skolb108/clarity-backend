import express from "express";

const app = express();

app.get("/", (req, res) => {
  res.send("Clarity backend running");
});

const PORT = process.env.PORT || 8080;

app.listen(PORT, "0.0.0.0", () => {
  console.log("Server running on", PORT);
});