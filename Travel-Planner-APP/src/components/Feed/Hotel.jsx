import React, { useEffect, useState } from "react";
import styled from "styled-components";
import { FaSearch as Search } from "react-icons/fa";
import axios from "axios";
import HotelGif from "../../assets/hotel.gif"; // Import the hotel GIF
import { getHotelRoute } from "../../utils/api-routes";

export default function Hotel() {
  const [text, setText] = useState("");
  const [hotel, setHotel] = useState([]);
  const user = JSON.parse(localStorage.getItem("user"));
  const [isS, setisS] = useState(false);

  const getHotel = async () => {
    setisS(true)
    try {
      const data = await axios.get(getHotelRoute, {
        headers: {
          Authorization: user.token,
        },
        params: {
          code: text,
        },
      });
      console.log(JSON.stringify(data.data));
      setHotel(data.data);
    } catch (e) {
      console.log(e.message);
    }
  };

  return (
    <Container>
      <div className="search">
        <Search className="icon" onClick={getHotel} />
        <input
          type="text"
          placeholder="Search Hotel"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>
      {isS ? (
        <div className="hotels">
          {hotel.map((hotel) => (
          <div className="hotel">
            <Card key={hotel.id} className="col-md-4">
              <div className="card-body">
                <CardTitle>{hotel.name}</CardTitle>
                <CardText>
                  <p>
                    Chain Code: <CardTextSpan>{hotel.chainCode}</CardTextSpan>
                  </p>
                  <p>
                    IATA Code: <CardTextSpan>{hotel.iataCode}</CardTextSpan>
                  </p>
                  <p>
                    Dupe ID: <CardTextSpan>{hotel.dupeId}</CardTextSpan>
                  </p>
                  <p>
                    Geo Code: <CardTextSpan>{`${hotel.geoCode.latitude} ${hotel.geoCode.longitude}`}</CardTextSpan>
                  </p>
                  <p>
                    Address: <CardTextSpan>{hotel.address.countryCode}</CardTextSpan>
                  </p>
                  <p>
                    Last Update: <CardTextSpan>{hotel.lastUpdate}</CardTextSpan>
                  </p>
                </CardText>
              </div>
            </Card>
          </div>
        ))}
        </div>
      ) : (
        <HotelGifContainer>
          <img src={HotelGif} alt="Hotel Loading" />
        </HotelGifContainer>
      )}
    </Container>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  overflow: hidden;
  border-right: #ffffff 4px;
  .search {
    border-radius: 0.5rem;
    background-color: #373842;
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 1rem;
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
  .hotels{
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    width: 80%;
    align-self: center;
    .hotel{
      padding: 0.5rem;
      border-radius: 2rem;
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

const CardTitle = styled.h5`
  color: #3875c9;
`;

const CardText = styled.div`
  color: #f8ecec;
  display: none;

  ${Card}:hover & {
    display: block;
  }
`;

const CardTextSpan = styled.span`
  color: #79d5f7;
  transition: color 0.3s;
  ${Card}:hover & {
    color: #e2e3f2;
  }
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

