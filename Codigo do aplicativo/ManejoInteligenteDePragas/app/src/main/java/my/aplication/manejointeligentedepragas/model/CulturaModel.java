package my.aplication.manejointeligentedepragas.model;

public class CulturaModel {
    private int Cod_Cultura;
    private int fk_Cod_Propriedade;
    private int fk_Cod_Planta;
    private String nomePlanta;
    private int numeroTalhoes;
    private double tamanhoCultura;
    private boolean aplicado;

    public CulturaModel(int cod_Cultura, int fk_Cod_Propriedade,int fk_Cod_Planta, String nomePlanta, int numeroTalhoes, boolean aplicado) {
        this.Cod_Cultura = cod_Cultura;
        this.fk_Cod_Propriedade = fk_Cod_Propriedade;
        this.fk_Cod_Planta = fk_Cod_Planta;
        this.nomePlanta = nomePlanta;
        this.numeroTalhoes = numeroTalhoes;
        this.aplicado = aplicado;
    }

    public CulturaModel(int cod_Cultura, int fk_Cod_Propriedade, int fk_Cod_Planta, String nomePlanta, int numeroTalhoes, double tamanhoCultura, boolean aplicado) {
        this.Cod_Cultura = cod_Cultura;
        this.fk_Cod_Propriedade = fk_Cod_Propriedade;
        this.fk_Cod_Planta = fk_Cod_Planta;
        this.nomePlanta = nomePlanta;
        this.numeroTalhoes = numeroTalhoes;
        this.tamanhoCultura = tamanhoCultura;
        this.aplicado = aplicado;
    }

    public CulturaModel() {
        this.Cod_Cultura = 0;
        this.fk_Cod_Propriedade = 0;
        this.fk_Cod_Planta = 0;
        this.nomePlanta = null;
        this.numeroTalhoes = 0;
        this.tamanhoCultura = 0;
        this.aplicado = false;
    }

    public int getCod_Cultura() {
        return Cod_Cultura;
    }

    public void setCod_Cultura(int cod_Cultura) {
        this.Cod_Cultura = cod_Cultura;
    }

    public int getFk_Cod_Propriedade() {
        return fk_Cod_Propriedade;
    }

    public void setFk_Cod_Propriedade(int fk_Cod_Propriedade) {
        this.fk_Cod_Propriedade = fk_Cod_Propriedade;
    }

    public String getnomePlanta() {
        return nomePlanta;
    }

    public void setnomePlanta(String nomePlanta) {
        this.nomePlanta = nomePlanta;
    }

    public int getNumeroTalhoes() {
        return numeroTalhoes;
    }

    public void setNumeroTalhoes(int numeroTalhoes) {
        this.numeroTalhoes = numeroTalhoes;
    }

    public double getTamanhoCultura(){return tamanhoCultura;}

    public void setTamanhoCultura(double tamanhoCultura){ this.tamanhoCultura = tamanhoCultura;}

    public boolean isAplicado() {
        return aplicado;
    }

    public void setAplicado(boolean aplicado) {
        this.aplicado = aplicado;
    }

    public int getFk_Cod_Planta() {
        return fk_Cod_Planta;
    }

    public void setFk_Cod_Planta(int fk_Cod_Planta) {
        this.fk_Cod_Planta = fk_Cod_Planta;
    }

}
