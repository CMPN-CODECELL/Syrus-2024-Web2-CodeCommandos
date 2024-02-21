const mongoose = require('mongoose')
const {tripSchema} = require('../models/TripModel')
const userSchema = mongoose.Schema({
    name: {
        type: String,
        required: true,
        min: 3,
        max: 15,
    },
    picture : {
        type: String
    },
    username : {
        type: String,
        required: true,
        min: 3,
        max: 15,
        unique: true
    },
    email : {
        type: String,
        required: true,
        min: 3,
        max: 45,
        unique: true,
        validate: {
            validator: function (value) {
                return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
            },
            msg: 'Invalid email format'
        }
    },
    password : {
        type: String,
        required: true,
        min: 8,
    },
    trips : {
        type: [tripSchema],
        default: []
    },
    tripRequest : {
        type: [mongoose.Types.ObjectId],
        default: []
    },
    posts: {
        type: [Object],
        default: []
    }
})

module.exports = mongoose.model("Users",userSchema)