const express = require("express");
const path = require('path')
const cors = require("cors");
const mongoose = require("mongoose");
const userRoutes = require("./routes/UserRoutes");
const getRoutes = require("./routes/AmadeusRoutes")
const tripRoutes = require('./routes/TripRoutes')
const suggRoutes = require('./routes/SuggestionRoutes')
const app = express();
app.use(cors());
app.use(express.json({limit: '50mb'}));
app.get("/",(_,res)=>res.json("WELCOME TO TRAVEL PLANNER APP"))
app.use("/api/auth",userRoutes);
app.use("/get/amadeus",getRoutes)
app.use("/trip",tripRoutes)
app.use("/sugg",suggRoutes)
app.use("/data/tickets",express.static(path.join("data","tickets")))
mongoose
  .connect(
    "mongodb+srv://extratdh:qwertyuiop@cluster0.te1qmaj.mongodb.net/"
  )
  .then(() => {
    console.log("DB Connected");
  })
  .catch((err) => {
    console.log(err.message);
  });

const server = app.listen(5000, () => {
  console.log(`Listening to 5000`);
})
