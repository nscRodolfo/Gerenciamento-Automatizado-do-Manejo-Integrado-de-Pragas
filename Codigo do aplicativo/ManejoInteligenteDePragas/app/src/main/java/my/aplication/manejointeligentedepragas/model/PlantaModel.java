package my.aplication.manejointeligentedepragas.model;

public class PlantaModel {
    private int Cod_Planta;
    private String Familia;
    private String Temperatura;
    private String PH;
    private String Espacamento;
    private String Solo;
    private String Nome;
    private String NomeCientifico;
    private String TamanhoTalhao;

    public PlantaModel(int cod_Planta, String familia, String temperatura, String PH, String espacamento, String solo, String nome, String nomeCientifico, String tamanhoTalhao) {
        this.Cod_Planta = cod_Planta;
        this.Familia = familia;
        this.Temperatura = temperatura;
        this.PH = PH;
        this.Espacamento = espacamento;
        this.Solo = solo;
        this.Nome = nome;
        this.NomeCientifico = nomeCientifico;
        this.TamanhoTalhao = tamanhoTalhao;
    }

    public int getCod_Planta() {
        return Cod_Planta;
    }

    public void setCod_Planta(int cod_Planta) {
        this.Cod_Planta = cod_Planta;
    }

    public String getFamilia() {
        return Familia;
    }

    public void setFamilia(String familia) {
        this.Familia = familia;
    }

    public String getTemperatura() {
        return Temperatura;
    }

    public void setTemperatura(String temperatura) {
        this.Temperatura = temperatura;
    }

    public String getPH() {
        return PH;
    }

    public void setPH(String PH) {
        this.PH = PH;
    }

    public String getEspacamento() {
        return Espacamento;
    }

    public void setEspacamento(String espacamento) {
        this.Espacamento = espacamento;
    }

    public String getSolo() {
        return Solo;
    }

    public void setSolo(String solo) {
        this.Solo = solo;
    }

    public String getNome() {
        return Nome;
    }

    public void setNome(String nome) {
        this.Nome = nome;
    }

    public String getNomeCientifico() {
        return NomeCientifico;
    }

    public void setNomeCientifico(String nomeCientifico) {
        this.NomeCientifico = nomeCientifico;
    }

    public String getTamanhoTalhao() {
        return TamanhoTalhao;
    }

    public void setTamanhoTalhao(String tamanhoTalhao) {
        this.TamanhoTalhao = tamanhoTalhao;
    }
}
