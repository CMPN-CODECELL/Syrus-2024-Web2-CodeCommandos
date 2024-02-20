import React from 'react';
import styled from 'styled-components';

const TripDetail = ({ tripData }) => {
  const { departureDate, departureTime, arrivalTime, tripMates, imageUrl, weather } = tripData;

  return (
    <Container>
      <DateAndTime>
        <p>Departure Date: {departureDate}</p>
        <p>Departure Time: {departureTime}</p>
        <p>Arrival Time: {arrivalTime}</p>
      </DateAndTime>
      
      <TripMates>
        <p>Trip Mates:</p>
        <ul>
          {tripMates.map((mate, index) => (
            <li key={index}>{mate}</li>
          ))}
        </ul>
      </TripMates>

      <Image src={imageUrl} alt="Trip Image" />

      <Weather>
        <p>Weather: {weather}</p>
      </Weather>
    </Container>
  );
};

const Container = styled.div`
  border: 1px solid #ccc;
  padding: 20px;
  margin: 10px;
  border-radius: 8px;
`;

const DateAndTime = styled.div`
  margin-bottom: 10px;
`;

const TripMates = styled.div`
  margin-bottom: 10px;

  ul {
    list-style: none;
    padding: 0;
  }
`;

const Image = styled.img`
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  margin-bottom: 10px;
`;

const Weather = styled.div``;

export default TripDetail;
