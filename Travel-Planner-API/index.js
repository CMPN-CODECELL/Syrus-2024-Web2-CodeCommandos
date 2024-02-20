const express = require("express");
const http = require("http");
const socketIO = require("socket.io");
const path = require("path");
const cors = require("cors");
const mongoose = require("mongoose");
const userRoutes = require("./routes/UserRoutes");
const getRoutes = require("./routes/AmadeusRoutes");
const tripRoutes = require("./routes/TripRoutes");
const suggRoutes = require("./routes/SuggestionRoutes");
const msgsRoutes = require("./routes/MessageRoutes");

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.get("/", (_, res) => res.json("WELCOME TO TRAVEL PLANNER APP"));
app.use("/api/msgs", msgsRoutes);
app.use("/api/auth", userRoutes);
app.use("/get/amadeus", getRoutes);
app.use("/trip", tripRoutes);
app.use("/sugg", suggRoutes);
app.use("/data/tickets", express.static(path.join("data", "tickets")));

mongoose
  .connect("mongodb+srv://extratdh:qwertyuiop@cluster0.te1qmaj.mongodb.net/")
  .then(() => {
    console.log("DB Connected");
  })
  .catch((err) => {
    console.log(err.message);
  });

io.on("connection", (socket) => {
  console.log("A user connected");

  socket.on("join room", (roomId) => {
    socket.join(roomId);
  });

  socket.on("chat message", ({ message, roomId }) => {
    io.to(roomId).emit("chat message", message);
  });

  socket.on("disconnect", () => {
    console.log("User disconnected");
  });
});

const PORT = process.env.PORT || 5000;

server.listen(PORT, () => {
  console.log(`Listening to ${PORT}`);
});
