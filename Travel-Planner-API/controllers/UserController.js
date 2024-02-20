const User = require("../models/UserModel");
const jwt = require("jsonwebtoken");
const jwtkey = "travelapi";
const bcrypt = require("bcrypt");

module.exports.register = async (req, res, next) => {
  try {
    const { name,username, email, password } = req.body;
    const isUsername = await User.findOne({ username });
    if (isUsername)
      return res.status(401).json({ msg: "Username already exists", status: false });
    const isEmail = await User.findOne({ email });
    if (isEmail) return res.status(401).json({ msg: "Email already used", status: false });
    const hashPass = await bcrypt.hash(password, 10);
    const user = await User.create({ name, username, email, password: hashPass });
    delete user.password;
    const token = jwt.sign({ user: user._id }, jwtkey);
    return res.json({ status: true,user,token});
  } catch (e) {
    res.status(401).json({msg: e.message, status: false})
  }
};
module.exports.login = async (req, res, next) => {
  try {
    const { username, password } = req.body;
    const user = await User.findOne({ username });
    if (!user) return res.status(401).json({ msg: "No user found", status: false });
    if (!(await bcrypt.compare(password, user.password)))
      return res.status(401).json({ msg: "Wrong Password", status: false });
    else {
      const token = jwt.sign({ user: user._id }, jwtkey);
      console.log(token);
      return res.json({ status: true,user,token});
    }
  } catch (e) {
    next(e);
  }
};
module.exports.updateProfile = async (req,res,next)=>{
  try {
    const filename = req.file.filename;
    console.log(req.body);
  } catch (error) {
    
  }
}
module.exports.getData = async (req,res,next) =>{
  try {
    const {user} = req.user
    const {username,name,email,trips} = await User.findOne({_id: user})
    return res.json({username,name,email,trips})
  } catch (e) {
    next(e)
  }
}
module.exports.getUsers = async (req,res,next) => {
  try {
    const {user} = req.user
    const users = await User.find({})
    const names = users.map((item)=>{
      return{
        _id : item._id,
        name: item.name
      }
    })
    const filtered = names.filter((item)=>{
      return item._id!=user
    })
    return res.json(filtered)
  } catch (e) {
    next(e)
  }
}
