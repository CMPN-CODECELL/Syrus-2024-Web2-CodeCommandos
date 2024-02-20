const Messages = require("../models/MessageModel");
module.exports.addMessage = async (req, res, next) => {
  try {
    console.log("b");
    const { msg,trip } = req.body;
    console.log(msg,trip);
    const {user} = req.user
    console.log(user);
    const data = await Messages.create({
      message: { text: msg },
      sender: user,
      trip
    });
    if (data) return res.json({ msg: "Message Added Successfully" });
    return res.json({ msg: "Message not added" });
  } catch (e) {
    next(e);
  }
};
module.exports.getAllMesssages = async (req, res, next) => {
  try {
    const {user} = req.user
    const { trip } = req.body;
    const messages = await Messages.find({
        trip: trip
      }).sort({ updatedAt: 1 });
    const projectMessage = messages.map((msg) => {
      return {
        fromSelf: msg.sender.toString() === user,
        message: msg.message.text,
      };
    });
    console.log(projectMessage);
    res.json(projectMessage);
  } catch (e) {
    next(e);
  }
};
