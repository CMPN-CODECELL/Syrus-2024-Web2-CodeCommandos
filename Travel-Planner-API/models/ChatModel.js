const mongoose = require("mongoose");
const {tripSchema} = require('../models/TripModel')
const trip = new mongoose.Schema({
    detail: {
        type: mongoose.Types.ObjectId
    },
    message: {
        text: {
          type: String,
          required: true,
        },
      },
      users: Array,
      sender: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true,
      },
})
const chatSchema = mongoose.Schema(
  {
    tripRequests: {
        type: [tripSchema]
    },
    trips: {
        type: [trip]
    }
  },
  {
    timestamps: true,
  }
);
module.exports = mongoose.model("Chats", chatSchema);
