const Suggestion = require("../models/SuggestionModel");
const axios = require("axios");
const User = require("../models/UserModel");
module.exports.addSugg = async (req, res, next) => {
  try {
    const { user } = req.user;
    const { tag, city, country, msg } = req.body;
    const body = { tag, loc: { city, country }, msg };
    try {
      const { data } = await axios.post("http://127.0.0.1:5000", { text: msg });
      const suggestion = Suggestion.create({...body,feedback: data.sentiment})
      console.log(suggestion);
      return res.json(suggestion)
    } catch (error) {
      console.log(error.message);
    }
  } catch (e) {
    console.log(e.message);
    res
      .status(500)
      .json({ status: false, msg: `Suggestion Not Added ${e.message}` });
    next(e);
  }
};
module.exports.removeSugg = async (req, res, next) => {
  try {
    const { suggId } = req.query;
    const sugg = await Suggestion.findByIdAndDelete({ _id: suggId });
    if (!sugg) return res.json({ status: false, msg: "No Suggestion Found" });
    return res.json({ status: true, msg: "Suggestion Removed" });
  } catch (e) {
    next(e);
  }
};
module.exports.getSugg = async (req, res, next) => {
  try {
    const { tag, city} = req.body;
    console.log(tag,city);
    const suggestion = await Suggestion.find({tag,'loc.city': city})
    console.log(suggestion);
  
    return res.json(suggestion)
  } catch (error) {
    next(e);
  }
};
module.exports.getUserSugg = async (req, res, next) => {
  try {
    const { user } = req.user;
    const { name } = await User.findOne({ _id: user });
    const suggestions = await Suggestion.find({ by: name });
    const mySuggestion = suggestions.map((item) => {
      const { createdAt } = item;
      const date = createdAt.toISOString().split("T")[0];
      const fname = item.by.split(" ")[0];
      return {
        id: item._id,
        by: fname,
        tag: item.tag,
        loc: {
          city: item.loc.city,
          country: item.loc.country,
        },
        msg: item.msg,
        votes: item.votes,
        date,
      };
    });
    return res.json(mySuggestion);
  } catch (e) {
    console.log(e.message);
    next(e);
  }
};
module.exports.voteSugg = async (req, res, next) => {
  try {
    const { suggId } = req.query;
    const suggestion = await Suggestion.findOne({ _id: suggId });
    suggestion.votes += 1;
    await suggestion.save();
    return res.json({ status: true, msg: "Vote Added" });
  } catch (e) {
    next(e);
  }
};
module.exports.removeVote = async (req, res, next) => {
  try {
    const { suggId } = req.query;
    const suggestion = await Suggestion.findOne({ _id: suggId });
    suggestion.votes -= 1;
    await suggestion.save();
    return res.json({ status: true, msg: "Vote Removed" });
  } catch (e) {
    next(e);
  }
};
