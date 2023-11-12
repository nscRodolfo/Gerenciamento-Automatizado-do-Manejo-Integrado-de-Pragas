package my.aplication.manejointeligentedepragas.model;

public class TalhaoModel {
    private int Cod_Talhao;
    private String Nome;
    private boolean aplicado;
    private String NumeroDePlantas;
    private int  fk_Cod_Cultura;
    private int fk_Cod_Planta;
    private String dataProxContagem;

    public TalhaoModel(int cod_Talhao,String Nome, String numeroDePlantas, int fk_Cod_Cultura, int fk_Cod_Planta, boolean aplicado,String dataProxContagem) {
        this.Cod_Talhao = cod_Talhao;
        this.Nome = Nome;
        this.aplicado = aplicado;
        this.NumeroDePlantas = numeroDePlantas;
        this.fk_Cod_Cultura = fk_Cod_Cultura;
        this.fk_Cod_Planta = fk_Cod_Planta;
        this.dataProxContagem = dataProxContagem;
    }

    public TalhaoModel(){
        this.Cod_Talhao = 0;
        this.Nome = null;
        this.aplicado = false;
        this.NumeroDePlantas = null;
        this.fk_Cod_Cultura = 0;
        this.fk_Cod_Planta = 0;
        this.dataProxContagem = null;
    }

    public int getCod_Talhao() {
        return Cod_Talhao;
    }

    public void setCod_Talhao(int cod_Talhao) {
        this.Cod_Talhao = cod_Talhao;
    }

    public String getNumeroDePlantas() {
        return NumeroDePlantas;
    }

    public void setNumeroDePlantas(String numeroDePlantas) {
        this.NumeroDePlantas = numeroDePlantas;
    }

    public int getFk_Cod_Cultura() {
        return fk_Cod_Cultura;
    }

    public void setFk_Cod_Cultura(int fk_Cod_Cultura) {
        this.fk_Cod_Cultura = fk_Cod_Cultura;
    }

    public int getFk_Cod_Planta() {
        return fk_Cod_Planta;
    }

    public void setFk_Cod_Planta(int fk_Cod_Planta) {
        this.fk_Cod_Planta = fk_Cod_Planta;
    }

    public String getNome() {
        return Nome;
    }

    public void setNome(String nome) {
        Nome = nome;
    }

    public boolean isAplicado() {
        return aplicado;
    }

    public void setAplicado(boolean aplicado) {
        this.aplicado = aplicado;
    }

    public String getDataProxContagem() {
        return dataProxContagem;
    }

    public void setDataProxContagem(String dataProxContagem) {
        this.dataProxContagem = dataProxContagem;
    }
}
