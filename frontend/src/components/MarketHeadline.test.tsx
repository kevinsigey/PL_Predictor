import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MarketHeadline } from "./MarketHeadline";

const model = { home: 0.5, draw: 0.2, away: 0.3 };

describe("MarketHeadline", () => {
  it("headlines the market's top outcome with the model beside it", () => {
    render(<MarketHeadline home="Arsenal" away="Spurs" market={{ home: 0.4, draw: 0.25, away: 0.35 }} model={model} />);
    expect(screen.getByTestId("market-headline")).toHaveTextContent("MarketArsenal 40%");
    expect(screen.getByTestId("market-headline")).toHaveTextContent("Model Arsenal 50%");
  });
  it("says there is no market line, and invents none", () => {
    render(<MarketHeadline home="Arsenal" away="Spurs" market={null} model={model} />);
    expect(screen.getByTestId("market-headline")).toHaveTextContent("Model: Arsenal 50% · No market line");
    expect(screen.queryByText(/Market Arsenal/)).toBeNull();
  });
});
