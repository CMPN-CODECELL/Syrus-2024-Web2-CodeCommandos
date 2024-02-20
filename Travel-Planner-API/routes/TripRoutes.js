const {addTrip, getTrip, getWeather, cancelTrip} = require('../controllers/TripController')
const {verifyToken} = require("../middlewares/UserMiddleware")
const file = require('../middlewares/TripTicketUploadMiddleware')
const router = require('express').Router()
router.post("/addTrip",verifyToken,file.single('image'),addTrip)
router.get("/getTrip",verifyToken,getTrip)
router.get("/getWeather",verifyToken,getWeather)
router.get("/cancelTrip",verifyToken,cancelTrip)
module.exports = router