import React, { useEffect, useState } from "react";
import styled from "styled-components";
import { FaSearch as Search } from "react-icons/fa";
import axios from "axios";
import { getHotelRoute } from "../../utils/api-routes";
export default function Hotel() {
  const [text, setText] = useState("");
  const [hotel, setHotel] = useState([]);
  const user = JSON.parse(localStorage.getItem("user"));

  const getHotel = async () => {
    try {
      const data = await axios.get(getHotelRoute, {
        headers: {
          Authorization: user.token,
        },
        params: {
          code: text, // Use the input text for the query parameter
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
      <div className="hotels">
      {hotel.map((hotel) => {
        console.log(hotel.name);
        return (
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
        );
      })}
      </div>
    </Container>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  border-right: #ffffff 4px;
  border-radius: 3rem;
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
