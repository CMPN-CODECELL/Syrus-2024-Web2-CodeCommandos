import React, { useEffect, useState } from "react";
import styled from "styled-components";
import axios from "axios";
import { FaVoicemail as Email } from "react-icons/fa";
import { getTrip, getUsers } from "../../utils/api-routes";
import { Link } from "react-router-dom";
import AddPost from "./AddPost";
import Post from "./Post";
export default function ShowTrip({ setAddShowTrip }) {
  const [posts, setPosts] = useState([]);
  const [trips, setTrips] = useState([]);
  const [weather, getWeather] = useState([]);
  const [add, setAdd] = useState(true);
  const user = JSON.parse(localStorage.getItem("user"));
  useEffect(() => {
    async function friends() {
      try {
        console.log("Fetching friends");
        const { data } = await axios.get(getUsers, {
          headers: {
            Authorization: user.token,
          },
        });
        const posts = data.map((post) => {
          if(post.posts && post.posts.length> 0){
            const array = post.posts.map((item)=>{
              return {name: post.name,imgs: item.imgs,caption: item.caption}
            })
            console.log(array);
            setPosts(array)
          }
        });
      } catch (error) {
        console.log(error.message);
      }
    }
    friends();
  }, []);
  useEffect(() => {
    async function getTrips() {
      try {
        const { data } = await axios.get(getTrip, {
          headers: {
            Authorization: user.token,
          },
        });
        setTrips(data);
      } catch (error) {
        console.log(error.message);
      }
    }
    getTrips();
  }, []);
  const handleAIButtonClick = () => {
    // Redirect to the AI link
    window.location.href =
      "https://travel-email-sender-7qklh6qvjevt2nkysgs7rj.streamlit.app/";
  };
  const handleEmailButtonClick = () => {
    window.location.href =
      "https://codecommandos-recommendtationsystem.netlify.app/";
  };
  return (
    <Container>
      <div className="add"></div>
      <div className="row">
        <div className="mytrips">
          {trips &&
            trips.map((item) => (
              <>
                <div className={`${item ? "trip" : "notrip"}`}>
                  <div className="tripdetail">
                    <div className="det">
                      <h4>{item.departureDate}</h4>
                      <h4>{item.returnDate}</h4>
                      <h4>{item.arrivalCity}</h4>
                      <img src={`http://localhost:5000${item.ticket}`} alt="" />
                      {item.tripMates.map((item) => (
                        <h4>{item}</h4>
                      ))}
                    </div>
                    <div className="weather">
                      <div className="search">
                        <h4>Get Weather</h4>
                      </div>
                      <div className="updates">
                        {[1, 4].map(() => (
                          <div class="card">
                            <div class="card-image">
                              <img src="whether.jpg" alt="Weather Image" />
                            </div>
                            <div class="card-content">
                              <h2>Weather Details</h2>
                              <div class="weather-details">
                                <h3>
                                  <span>Date:</span>
                                  <b class="datee"> 20th February 2024</b>
                                </h3>
                                <p>Time: 12:00 PM</p>
                                <p>Temperature: 25°C</p>
                                <p>Clouds: Partly Cloudy</p>
                                <p>Pressure: 1013 hPa</p>
                                <p>Sea Level: 0 m</p>
                                <p>Ground Level: 0 m</p>
                                <p>Wind Speed: 10 km/h</p>
                                <p>Wind Gust: 15 km/h</p>
                                <p>Wind Degree: 180°</p>
                                <p>Humidity: 70%</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ))}
        </div>
        <div className="myposts">
          {add ? (
            <AddPost setAdd={setAdd} />
          ) : (
            <div className="posts">
              {posts.map((item) => {
                if (item) {
                  return <Post item={item} />;
                }
              })}
            </div>
          )}
        </div>
      </div>
      <FloatingActionButton3 onClick={handleEmailButtonClick}>
        <Email />
      </FloatingActionButton3>
      <FloatingActionButton2 onClick={handleAIButtonClick}>
        AI
      </FloatingActionButton2>
      <FloatingActionButton onClick={() => setAddShowTrip("add")}>
        +
      </FloatingActionButton>
    </Container>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  .row {
    height: 100%;
    display: grid;
    grid-template-columns: 50% 50%;
    .mytrips {
      height: 100%;
      width: 100%;
      .trip {
        height: 100%;
        .tripdetail {
          height: 100%;
          display: grid;
          grid-template-columns: 50% 50%;
          background-color: #282a34;
          border-radius: 2rem;
          padding: 1rem;
          .det {
            h4 {
              color: white;
            }
            img {
              align-self: center;
            }
          }
          .weather {
            display: none;
            h4 {
              color: white;
            }
            .updates {
              display: flex;
              flex-direction: row;
              flex: 0 0 10%;
              .card {
              }
            }
          }
          .notrip {
            display: none;
          }
        }
      }
    }
    .myposts {
      height: 100%;
      width: 100%;
      overflow: auto;
      &-thumb {
        background-color: #ffffff39;
        width: 0.1rem;
        border-radius: 1rem;
      }
    }
  }
`;
const FloatingActionButton = styled.button`
  position: fixed;
  bottom: 20px;
  right: 20px;
  background-color: #61dafb;
  color: white;
  padding: 15px;
  border: none;
  font-size: 1.5em;
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease-in-out;

  &:hover {
    background-color: #4e0eff;
  }
`;
const FloatingActionButton2 = styled.button`
  position: fixed;
  bottom: 80px;
  right: 20px;
  background-color: #61dafb;
  color: white;
  padding: 15px;
  border: none;
  font-size: 1.5em;
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease-in-out;

  &:hover {
    background-color: #4e0eff;
  }
`;
const FloatingActionButton3 = styled.button`
  position: fixed;
  bottom: 140px;
  right: 20px;
  background-color: #61dafb;
  color: white;
  padding: 15px;
  border: none;
  font-size: 1.5em;
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease-in-out;

  &:hover {
    background-color: #4e0eff;
  }
`;
