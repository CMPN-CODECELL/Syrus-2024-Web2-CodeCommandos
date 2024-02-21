import React, {useState } from "react";
import styled from "styled-components";
import { FaSearch as Search } from "react-icons/fa";
import axios from "axios";
import { getFightRoute } from "../../utils/api-routes";
import flightimg from '../../assets/flight.gif'
export default function Flight() {
  const [isS,setisS] = useState(false)
  const [src, setSrc] = useState("");
  const [des, setDes] = useState("");
  const [flight, setFlight] = useState([]);
  const user = JSON.parse(localStorage.getItem("user"));
 const handleClick = async()=>{
  setisS(true)
      try {
        const { data } = await axios.get(getFightRoute, {
          headers: {
            Authorization: user.token,
          },
          params: {
            src: "DEL",
            des: "BOM",
          },
        });
        console.log(JSON.stringify(data));
        setFlight(data);
      } catch (e) {
        console.log(e.message);
      }
    }

  return (
    <Container>
      <div className="top">
        <div className="search">
          <input
            type="text"
            placeholder="Source Airport"
            value={src}
            onChange={(e) => setSrc(e.target.value)}
          />
        </div>
        <div className="search">
          <Search className="icon"  onClick={handleClick} />
          <input
            type="text"
            placeholder="Destination Airport"
            value={des}
            onChange={(e) => setDes(e.target.value)}
          />
        </div>
      </div>
      {
        isS ? (
      <div className="flights">
        {flight.map((flight, index) => {
          return (
            <div className="flight">
              <Card key={index} className="col-md-4">
                <div className="card-body">
                  <CardTitle>{flight.date}</CardTitle>
                  <div className="rowf">
                    {flight.flightData.map((item) => {
                      const { duration, price, departure, arrival } = item;
                      const { iataCode: di, terminal: dt, at: da } = departure;
                      const { iataCode: ai, terminal: at, at: aa } = arrival;
                      return (
                        <Card2>
                          <div className="card-body" key={index}>
                            <div className="top">
                              <h4>{duration}</h4>
                              <h4>{price.toFixed(2)}</h4>
                            </div>
                            <div className="detail">
                              <h4>
                                {`${di} ${dt} ${da}`}
                              </h4>
                              <h4>
                                {`${ai} ${at} ${aa}`}
                              </h4>
                            </div>
                          </div>
                        </Card2>
                      );
                    })}
                  </div>
                </div>
              </Card>
            </div>
          );
        })}
      </div>): (
        <HotelGifContainer>
          <img src={flightimg} alt="" />
        </HotelGifContainer>
      )}
    </Container>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  .top {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    .search {
      border-radius: 0.5rem;
      background-color: #373842;
      display: flex;
      flex-direction: row;
      justify-content: center;
      gap: 2rem;
      width: fit-content;
      input {
        outline: none;
        padding: 0.5rem;
        background-color: transparent;
        color: #7e7f89;
        border: none;
      }
      .icon {
        align-self: center;
        color: #595a63;
      }
    }
  }
  .flights {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    width: 100%;
    align-self: center;
    .flight {
      height: 100%;
      width: 100%;
      padding: 0.5rem;
      border-radius: 2rem;
      .card-body {
        width: 100%;
        .rowf {
          width: 100%;
          flex-wrap: wrap;
          width: 100%;
          display: flex;
          flex-direction: row;
        }
      }
    }
  }
`;
const Card = styled.div`
  background-color: #282a34;
  color: #fff;
  padding: 0.5rem;
  border-radius: 0.5rem;
  box-shadow: 0 0 3px 3px #ffffff;
  transition: transform 0.3s, box-shadow 0.3s;
  cursor: pointer;
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
`;
const Card2 = styled.div`
  background-color: #282a34;
  color: #fff;
  padding: 0.5rem;
  border-radius: 0.5rem;
  transition: transform 0.3s, box-shadow 0.3s;
  cursor: pointer;
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
`;
const CardTitle = styled.h3`
  color: #3875c9;
`;
const HotelGifContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  width: 100%;
  img{
    align-self: center;
    height: 80%;
    width: 80%;
  }
`;

