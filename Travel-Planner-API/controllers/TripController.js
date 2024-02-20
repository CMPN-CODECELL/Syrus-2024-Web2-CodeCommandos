const { default: axios } = require("axios");
const fs = require('fs')
const User = require("../models/UserModel");
const { Trip } = require("../models/TripModel");
const { log } = require("util");
module.exports.addTrip = async (req, res, next) => {
  try {
    const id = req.user.user;
    const { departureDate, arrivalTime, returnDate, arrivalCity,tripMates } = req.body;
    const filename = req.file.filename;
    const imageUrl = `/data/tickets/${filename}`;
    const mates = JSON.parse(tripMates)
    const user = await User.findOne({ _id: id });
    const obj = {
      departureDate,departureTime: arrivalTime,returnDate,arrivalCity,tripMates: mates,ticket: imageUrl
    }
    console.log(obj);
    let trip = new Trip(obj)
    await trip.save()
    user.trips.unshift(trip);
    await user.save();
    return res.json({ msg: "Trip Added" });
  } catch (e) {
    if (req.file && req.file.filename) {
      try {
        fs.unlink(`data/tickets/${req.file.filename}`,()=>console.log("File Deleted"));
      } catch (err) {
        console.error('Error deleting file:', err);
      }
    }
    next(e)
  }
};
module.exports.getTrip = async (req, res, next) => {
  try {
    console.log("ou");
    const id = req.user.user;
    const user = await User.findOne({ _id: id });
    const trips = user.trips;
    return res.json(trips);
  } catch (e) {
    next(e);
  }
}
module.exports.cancelTrip = async (req,res,next)=>{
  try {
    const {id} = req.query
    await Trip.findOneAndDelete({_id: id})
    const iD = req.user.user
    const user = await User.findOne({_id: iD})
    let userTrips = user.trips
    user.trips = userTrips.filter((item)=>{
      if(item._id.toString()===id) return false
      else return true
    })
    console.log(userTrips);
    await user.save()
    return res.json({msg: "Trip Cancelled"})
  } catch (e) {
    next(e)
  }
}
module.exports.getWeather = async (req, res, next) => {
  try {
    const { cityName } = req.query;
    console.log(cityName);
    const openWeatherAPIKey = "52363e6c0c51824d1e3dd833dc3c0918";
    const url = `https://api.openweathermap.org/data/2.5/forecast?q=${cityName.trim()},&APPID=${openWeatherAPIKey}`;
    try {
      console.log(url);
      const weather = await axios.get(url);
      const hour = weather.data.list;
      const data = hour.map((item) => {
        const { main, weather, clouds, wind, dt_txt } = item;
        const { temp, pressure, humidity, sea_level, grnd_level } = main;
        const tempinC = temp - 273.15;
        const { description, icon } = weather[0];
        const iconurl = `https://openweathermap.org/img/wn/${icon}@2x.png`;
        const { all } = clouds;
        const { speed, deg, gust } = wind;
        return {
          tempinC,
          pressure,
          humidity,
          sea_level,
          grnd_level,
          description,
          iconurl,
          cloud: all,
          windSpeed: speed,
          windDeg: deg,
          windGust: gust,
          time: dt_txt,
        };
      });
      return res.json(data);
    } catch (e) {
      console.log(e.message);
      throw e;
    }
  } catch (e) {
    console.log(e.message);
    next(e);
  }
}