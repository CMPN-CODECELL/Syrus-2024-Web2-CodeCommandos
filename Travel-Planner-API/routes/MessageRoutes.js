const {addMessage,getAllMesssages} = require('../controllers/MessageController')
const {verifyToken} = require('../middlewares/UserMiddleware')
const router = require('express').Router()
router.post("/addmsg",verifyToken,addMessage)
router.post("/getmsgs",verifyToken,getAllMesssages)
module.exports = router