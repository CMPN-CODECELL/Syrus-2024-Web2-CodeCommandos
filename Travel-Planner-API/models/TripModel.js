const mongoose = require('mongoose')
const tripSchema = mongoose.Schema({
    departureDate : {
        type: String,
        required: true,
    },
    returnDate : {
        type: String,
        required: true,
    },
    arrivalCity : {
        type: String,
        required: true
    },
    ticket : {
        type: String,
        required: true
    },
    tripMates: {
        type : [mongoose.Types.ObjectId],
        default: []
    }
})
const Trip = mongoose.model("Trips",tripSchema)
module.exports = {Trip,tripSchema}