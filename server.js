import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.status(200).send("Clarity backend running");
});

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/api/chat", (req, res) => {
  res.json({
    reply: "Backend working"
  });
});

const PORT = process.env.PORT || 8080;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server listening on ${PORT}`);
});