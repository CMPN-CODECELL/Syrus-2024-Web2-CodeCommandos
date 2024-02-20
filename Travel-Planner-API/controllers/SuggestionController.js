const Suggestion = require("../models/SuggestionModel");
const User = require("../models/UserModel");
module.exports.addSugg = async (req, res, next) => {
  try {
    const { user } = req.user;
    console.log(user);
    const { name } = await User.findOne({ _id: user });
    console.log(name);
    const { tag, city, country, msg } = req.query;
    console.log(tag, city, country, msg);
    const body = { by: name, tag, loc: { city, country }, msg };
    await Suggestion.create(body);
    console.log("fu");
    return res.json({ status: true, msg: "Suggestion Added" });
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
    const { tag, city, country } = req.query;
    if (tag && city) {
      const suggestions = await Suggestion.findOne({ "loc.city": city });
      const { name } = await User.findOne({ _id: suggestions.by });
      return res.json(suggestions);
    } else {
      const suggestions = await Suggestion.findOne({ "loc.country": country });
      return res.json(suggestions);
    }
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
module.exports.voteSugg = async (req,res,next)=>{
  try {
    const {suggId} = req.query
    const suggestion = await Suggestion.findOne({_id: suggId})
    suggestion.votes += 1
    await suggestion.save()
    return res.json({ status: true, msg: "Vote Added" });
  } catch (e) {
    next(e)
  }
}
module.exports.removeVote = async (req,res,next)=>{
  try {
    const {suggId} = req.query
    const suggestion = await Suggestion.findOne({_id: suggId})
    suggestion.votes -= 1
    await suggestion.save()
    return res.json({ status: true, msg: "Vote Removed" });
  } catch (e) {
    next(e)
  }
}