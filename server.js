import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();

/* Middleware */
app.use(cors());
app.use(express.json());

/* Root route */
app.get("/", (req, res) => {
  res.status(200).send("Clarity backend running");
});

/* Health check */
app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

/* Test endpoint */
app.post("/api/chat", (req, res) => {
  res.json({
    reply: "Backend works"
  });
});

/* IMPORTANT PART */

const PORT = process.env.PORT;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server listening on ${PORT}`);
});