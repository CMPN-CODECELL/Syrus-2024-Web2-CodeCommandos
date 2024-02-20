import React, { useEffect, useState } from "react";
import styled from "styled-components";
import { FaSearch as Search } from "react-icons/fa";
import axios from "axios";
import { getFightRoute} from "../../utils/api-routes";
export default function Flight() {
  const [des, setDes] = useState("");
  const [src, setSrc] = useState("");
  const [flight, setFlight] = useState([]);
  const user = JSON.parse(localStorage.getItem("user"));
  console.log(user);
  useEffect(() => {
    async function getHotel() {
      try {
        const  data  = await axios.get(getFightRoute, {
          headers: {
            Authorization: user.token,
          },
          params: {
            code: "BOM",
          },
        });
        console.log(JSON.stringify(data.data));
        setFlight(data.data);
      } catch (e) {
        console.log(e.message);
      }
    }
    getHotel();
  }, []);
  return (
    <Container>
      <div className="top">
      <div className="search">
        <Search className="icon" />
        <input
          type="text"
          placeholder="Search Flight"
          value={src}
          onChange={(e) => setSrc(e.target.value)}
        />
      </div>
      <div className="search">
        <Search className="icon" />
        <input
          type="text"
          placeholder="Search Flight"
          value={src}
          onChange={(e) => setSrc(e.target.value)}
        />
      </div>
      </div>
      <div className="flights">

      {flight.map((flight) => {
        return (
          <div className="flight">
            <Card key={hotel.id} className="col-md-4">
              <div className="card-body">
                <CardTitle>{flight.date}</CardTitle>
                {
                  flight.map((item)=>{
                    return(
                      <Card/>
                    )
                  })
                }
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
  .top{
    display: flex;
    flex-direction: column;
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
