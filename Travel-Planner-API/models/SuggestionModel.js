const mongoose = require("mongoose");
const locSchema = new mongoose.Schema({
    city: {
        type: String,
        required: true
    },
    country: {
        type: String,
        required: true
    },
})
const suggestSchema = mongoose.Schema(
  {
    by: {
      type: String,
      required: true,
    },
    tag : {
        type: String,
        required: true,
    },
    msg : {
        type: String,
        required: true
    },
    loc : {
        type: locSchema
    },
    votes : {
      type : Number,
      default : 0
    }
  },
  {
    timestamps: true,
  }
);
module.exports = mongoose.model("Suggest", suggestSchema);

