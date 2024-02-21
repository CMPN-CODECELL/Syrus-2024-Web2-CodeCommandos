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
    feedback : {
      type: String
    }
  },
  {
    timestamps: true,
  }
);
module.exports = mongoose.model("Suggest", suggestSchema);

