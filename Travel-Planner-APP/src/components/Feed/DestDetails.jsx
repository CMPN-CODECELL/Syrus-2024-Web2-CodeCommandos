import React, { useRef } from "react";
import styled from "styled-components";
export const EventDetails = (props) => {
  const {
    name,
    amount,
    currencyCode,
    minimumDuration,
    bookingLink,
  } = props.eventData;

  const carouselRef = useRef(null);

  const scrollLeft = () => {
    if (carouselRef.current) {
      carouselRef.current.scrollBy({
        left: -300, 
        behavior: "smooth",
      });
    }
  };

  const scrollRight = () => {
    if (carouselRef.current) {
      carouselRef.current.scrollBy({
        left: 300, 
        behavior: "smooth",
      });
    }
  };

  return (
    <Container>
      <CarouselContainer>
        <Carousel ref={carouselRef}>
          {props.eventData.pictures.map((picture, index) => (
            <CarouselItem key={index}>
              <img src={picture} alt={`carousel-img-${index}`} />
            </CarouselItem>
          ))}
        </Carousel>
        <CarouselCaption>
          <h2>{name}</h2>
          <p>{`Amount: ${amount} ${currencyCode}`}</p>
          <p>{`Minimum Duration: ${minimumDuration}`}</p>
          <BookingLink
            href={bookingLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            Book Now
          </BookingLink>
        </CarouselCaption>
        <Button onClick={scrollLeft}>{"<"}</Button>
        <Button onClick={scrollRight}>{">"}</Button>
      </CarouselContainer>
    </Container>
  );
};
const CarouselItem = styled.div`
  flex: 0 0 100%;

  img {
    width: 100%;
    height: auto;
    object-fit: cover;
  }

  overflow-y: hidden;
`;
// ... (previous imports)

const Button = styled.button`
  position: absolute;
  top: 10%;
  transform: translateY(-50%);
  background: transparent;
  border: none;
  font-size: 1.5rem;
  color: #3c3f51;
  cursor: pointer;

  &:first-child {
    left: 0;
  }

  &:last-child {
    right: 0;
  }
`;
const Container = styled.div`
  width: 100%;
  height: 100%;
`;

const CarouselContainer = styled.div`
  position: relative;
`;

const Carousel = styled.div`
  display: flex;
  overflow-x: scroll;
  scroll-snap-type: x mandatory;

  &::-webkit-scrollbar {
    width: 0.5rem;
    &-thumb {
      background-color: #ffffff39;
      width: 0.1rem;
      border-radius: 1rem;
    }
  }

  & > ${CarouselItem} {
    scroll-snap-align: start;
  }
`;



const CarouselCaption = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  padding: 20px;
  text-align: center;
  background: rgba(0, 0, 0, 0.6);
  color: white;

  h2 {
    margin: 0;
  }

  p {
    margin: 5px 0;
  }
`;

const BookingLink = styled.a`
  top: 0.1rem;
  left: 1.5rem;
  position: absolute;
  display: block;
  margin-top: 20px;
  padding: 10px 20px;
  color: blue;
  text-align: center;
  text-decoration: none;
  border-radius: 5px;
`;

export default EventDetails;
