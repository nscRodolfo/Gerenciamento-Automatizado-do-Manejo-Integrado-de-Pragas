package my.aplication.manejointeligentedepragas.model;

public class AtingeModel {
    private int Cod_Atinge;
    private double NivelDanoEconomico;
    private double NivelDeControle;
    private double NivelDeEquilibrio;
    private int NumeroPlantasAmostradas;
    private int PontosPorTalhao;
    private int PlantasPorPonto;
    private int NumAmostras;

    private int fk_Cod_Planta;
    private int fk_Cod_Praga;

    public AtingeModel(int cod_Atinge, double nivelDanoEconomico, double nivelDeControle, double nivelDeEquilibrio, int numeroPlantasAmostradas, int pontosPorTalhao, int plantasPorPonto, int fk_Cod_Planta, int fk_Cod_Praga, int numAmostras) {
        this.Cod_Atinge = cod_Atinge;
        this.NivelDanoEconomico = nivelDanoEconomico;
        this.NivelDeControle = nivelDeControle;
        this.NivelDeEquilibrio = nivelDeEquilibrio;
        this. NumeroPlantasAmostradas = numeroPlantasAmostradas;
        this.PontosPorTalhao = pontosPorTalhao;
        this.PlantasPorPonto = plantasPorPonto;
        this.fk_Cod_Planta = fk_Cod_Planta;
        this.fk_Cod_Praga = fk_Cod_Praga;
        this.NumAmostras = numAmostras;
    }
    public AtingeModel() {
        this.Cod_Atinge = 0;
        this.NivelDanoEconomico = 0;
        this.NivelDeControle = 0;
        this.NivelDeEquilibrio = 0;
        this. NumeroPlantasAmostradas = 0;
        this.PontosPorTalhao = 0;
        this.PlantasPorPonto = 0;
        this.fk_Cod_Planta = 0;
        this.fk_Cod_Praga = 0;
        this.NumAmostras = 0;
    }

    public int getCod_Atinge() {
        return Cod_Atinge;
    }

    public void setCod_Atinge(int cod_Atinge) {
        this.Cod_Atinge = cod_Atinge;
    }

    public double getNivelDanoEconomico() {
        return NivelDanoEconomico;
    }

    public void setNivelDanoEconomico(double nivelDanoEconomico) {
        this.NivelDanoEconomico = nivelDanoEconomico;
    }

    public double getNivelDeControle() {
        return NivelDeControle;
    }

    public void setNivelDeControle(double nivelDeControle) {
        this.NivelDeControle = nivelDeControle;
    }

    public double getNivelDeEquilibrio() {
        return NivelDeEquilibrio;
    }

    public void setNivelDeEquilibrio(double nivelDeEquilibrio) {
        this.NivelDeEquilibrio = nivelDeEquilibrio;
    }

    public int getNumeroPlantasAmostradas() {
        return NumeroPlantasAmostradas;
    }

    public void setNumeroPlantasAmostradas(int numeroPlantasAmostradas) {
        this.NumeroPlantasAmostradas = numeroPlantasAmostradas;
    }

    public int getPontosPorTalhao() {
        return PontosPorTalhao;
    }

    public void setPontosPorTalhao(int pontosPorTalhao) {
        this.PontosPorTalhao = pontosPorTalhao;
    }

    public int getPlantasPorPonto() {
        return PlantasPorPonto;
    }

    public void setPlantasPorPonto(int plantasPorPonto) {
        this.PlantasPorPonto = plantasPorPonto;
    }

    public int getFk_Cod_Planta() {
        return fk_Cod_Planta;
    }

    public void setFk_Cod_Planta(int fk_Cod_Planta) {
        this.fk_Cod_Planta = fk_Cod_Planta;
    }

    public int getFk_Cod_Praga() {
        return fk_Cod_Praga;
    }

    public void setFk_Cod_Praga(int fk_Cod_Praga) {
        this.fk_Cod_Praga = fk_Cod_Praga;
    }

    public int getNumAmostras() {
        return NumAmostras;
    }

    public void setNumAmostras(int numAmostras) {
        NumAmostras = numAmostras;
    }
}
